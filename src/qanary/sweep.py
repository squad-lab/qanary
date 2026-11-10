import asyncio
import hashlib
import os
from math import isfinite
from numbers import Real
from threading import Event, Thread
from time import monotonic, sleep, time
from typing import Any, Callable, Optional, Sequence

import numpy as np
from qcodes.parameters import Parameter
from zarr.api.asynchronous import open_array
from zarr.core.sync import sync

from qanary.buffered.sweep import (
    _arm_instruments,
    _buffered_sweep_progress_info,
    _fetch_results,
)
from qanary.logger import get_logger

# `_stepper` and the other underscored helpers are internal.
__all__ = [
    "Sweep",
    "CircularSweep",
    "PointSweep",
    "SegmentedSweep",
    "sweeper",
    "rampdown",
    "reset_disk_persist_cache",
]


logger = get_logger(__name__)
last_save = 0

DISK_PERSIST_ATTEMPTS = 6
DISK_PERSIST_RETRY_DELAY = 0.2

# Keep live checkpoints compatible with stores written before Zarr 3 support.
ZARR_FORMAT = 2

# Cache the last digest of each disk-store key so checkpoints only write changes.
# The store path is stable across recursive `_stepper` calls.
_DISK_KEY_DIGESTS: dict[str, dict[str, bytes]] = {}


def _key_digest(value: Any) -> bytes:
    """
    Return a short digest of one store value.

    Args:
        value (Any): Encoded Zarr buffer.

    Returns:
        bytes: Digest used to detect changes between checkpoints.

    """
    data = np.ascontiguousarray(value.as_numpy_array())
    return hashlib.blake2b(data, digest_size=16).digest()


def _store_cache_key(store: Any) -> str:
    """
    Identify a store in the digest cache.

    Args:
        store (Any): Store object, or a path standing in for one.

    Returns:
        str: The normalized ``LocalStore`` root, or the string form of *store*.

    """
    return str(getattr(store, "root", store))


def reset_disk_persist_cache(store: Any = None) -> None:
    """
    Forget what a checkpoint believes is already on disk.

    Args:
        store (Any): Store, or its path, to forget. Forgets every store when
            omitted.

    """
    if store is None:
        _DISK_KEY_DIGESTS.clear()
        _MEMORY_FLUSHED_ROWS.clear()
    else:
        _DISK_KEY_DIGESTS.pop(_store_cache_key(store), None)


def _copy_changed_keys(source: dict, target: Any, store_path: str) -> int:
    """
    Copy keys that changed since the last checkpoint.

    Args:
        source (dict): In-memory store populated by Zarr.
        target (Any): Zarr store to write to, typically a ``LocalStore``.
        store_path (str): Identifies ``target`` in the digest cache.

    Returns:
        int: Number of keys written.

    Raises:
        PermissionError: If the target store is locked.

    """
    digests = _DISK_KEY_DIGESTS.setdefault(store_path, {})
    current = [(key, value, _key_digest(value)) for key, value in list(source.items())]
    return sync(_write_changed_keys(current, target, digests))


async def _write_changed_keys(
    current: list, target: Any, digests: dict[str, bytes]
) -> int:
    """
    Write the keys whose digest differs from the one last written.

    Args:
        current (list): ``(key, value, digest)`` for every key of the source.
        target (Any): Zarr store to write to.
        digests (dict[str, bytes]): Digests last written to ``target``,
            updated as writes land.

    Returns:
        int: Number of keys written.

    """
    # A missing group means the cached digests no longer describe this store.
    if digests and not await target.exists(".zgroup"):
        digests.clear()

    async def write(key: str, value: Any, digest: bytes) -> None:
        await target.set(key, value)
        # Update the digest only after a successful write so failures are retried.
        digests[key] = digest

    changed = [item for item in current if digests.get(item[0]) != item[2]]
    await _gather_all([write(*item) for item in changed])
    return len(changed)


async def _gather_all(coroutines: list) -> None:
    """
    Run every coroutine before raising the first error.

    Waiting for all writes prevents a retry from racing unfinished writes.

    Args:
        coroutines (list): Coroutines to run.

    """
    for result in await asyncio.gather(*coroutines, return_exceptions=True):
        if isinstance(result, BaseException):
            raise result


async def _write_rows(
    store: dict, arrays: dict, rows: dict[str, np.ndarray], low: int, high: int
) -> None:
    """
    Write rows ``low:high`` of each array in an in-memory store.

    Args:
        store (dict): The in-memory store.
        arrays (dict): Arrays already opened in ``store``, by name. Arrays
            opened here are added to it.
        rows (dict[str, np.ndarray]): New values of the rows, by array name.
        low (int): First row.
        high (int): Row after the last.

    """

    async def write(name: str, values: np.ndarray) -> None:
        if name not in arrays:
            arrays[name] = await open_array(
                store=store, path=name, mode="r+", zarr_format=ZARR_FORMAT
            )
        await arrays[name].setitem(slice(low, high), values)

    await _gather_all([write(name, values) for name, values in rows.items()])


def _copy_store(source: dict, target: Any) -> int:
    """
    Copy every key of an in-memory store into ``target`` and remember them.

    Args:
        source (dict): The in-memory store.
        target (Any): Zarr store to write to.

    Returns:
        int: Number of keys written.

    """
    reset_disk_persist_cache(target)
    return _copy_changed_keys(source, target, _store_cache_key(target))


# Track flushed rows by store identity across recursive `_stepper` calls.
_MEMORY_FLUSHED_ROWS: dict[int, int] = {}


def _memory_region(dataset: Any, position: Any, flushed: int) -> Any:
    """
    Return the row range to flush to the in-memory store.

    Args:
        dataset (Any): Dataset being written.
        position (Any): Coordinate values of the point just acquired, or
            ``None`` when the caller does not know where the sweep is.
        flushed (int): First row not yet known to be flushed.

    Returns:
        Any: ``(low, high, outer_dim)`` for a region write, or ``None`` when the
            dataset's shape makes one unsafe and the whole store must be
            rewritten.

    """
    variables = list(dataset.data_vars.values())
    if not variables or position is None:
        return None

    dims = variables[0].dims
    if not dims:
        return None
    outer = dims[0]
    # One region can cover every variable only when the outer dimension matches.
    if any(variable.dims[:1] != (outer,) for variable in variables):
        return None
    if outer not in position:
        return None

    # Circular sweeps are not monotonic. Repeated values make the row ambiguous,
    # so use a full write instead of guessing.
    rows = np.flatnonzero(dataset[outer].values == position[outer])
    if rows.size != 1:
        return None

    row = int(rows[0])
    return min(flushed, row), row + 1, outer


class _BufferedProgress:
    def __init__(self, bar, points: int, estimated_duration: float) -> None:
        """
        Initialize progress estimation for one buffered block.

        Args:
            bar (Any): Progress bar exposing an ``update`` method.
            points (int): Number of coordinate points in the buffered block.
            estimated_duration (float): Estimated duration of the block in
                seconds.

        Raises:
            TypeError: If ``points`` or ``estimated_duration`` cannot be
                converted to their declared numeric types.
            ValueError: If ``points`` or ``estimated_duration`` contains an
                invalid numeric value.

        """
        self.bar = bar
        self.points = max(1, int(points))
        self.estimated_duration = max(0.0, float(estimated_duration))
        self.advanced = 0
        self._stop_event = Event()
        self._thread = None

    def start(self) -> None:
        """
        Start estimated progress updates in a daemon thread.

        """
        # Completion of the final point means results were fetched and stored,
        # so the timer deliberately never advances that point.
        if self.points <= 1 or self.estimated_duration <= 0:
            return

        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        """
        Advance estimated progress without completing the final point.

        """
        started_at = monotonic()
        refresh_interval = min(0.1, max(0.01, self.estimated_duration / self.points))

        while not self._stop_event.wait(refresh_interval):
            elapsed = monotonic() - started_at
            target = min(
                self.points - 1,
                int(self.points * elapsed / self.estimated_duration),
            )
            if target > self.advanced:
                self.bar.update(target - self.advanced)
                self.advanced = target

    def stop(self) -> None:
        """
        Stop and join the progress-estimation thread if it was started.

        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

    def complete(self) -> None:
        """
        Stop estimation and advance through the final buffered point.

        """
        self.stop()
        if self.advanced < self.points:
            self.bar.update(self.points - self.advanced)
            self.advanced = self.points


class Sweep:
    def __init__(
        self,
        parameter: Parameter | Sequence[Parameter],
        start: int | float,
        stop: int | float,
        num: int | float = 0.0,
        step: int | float = 0.0,
        delay: float = 0.0,
        start_delay: float = 0.0,
        ramprate: float = 0.0,
        spacing: str = "lin",
    ) -> None:
        """
        Define an inclusive linear or logarithmic parameter sweep.

        Args:
            parameter (Parameter | Sequence[Parameter]): QCoDeS parameter or
                parameters that share the same setpoints.
            start (int | float): First setpoint.
            stop (int | float): Final setpoint.
            num (int | float): Number of points. Ignored when *step* is nonzero.
            step (int | float): Step-size request converted to
                ``int(abs(start - stop) / step)`` points.
            delay (float): Dwell time in seconds after each setpoint. Defaults
                to 0.
            start_delay (float): Delay in seconds after moving to the first
                setpoint. Defaults to 0.
            ramprate (float): If nonzero, replace *delay* with the time per
                point implied by this units-per-second rate.
            spacing (str): ``"lin"`` or ``"log"``. Defaults to ``"lin"``.

        Raises:
            ValueError: If neither *num* nor *step* is provided, or if *spacing*
                is not supported.

        """
        if not (step or num):
            logger.error("Either one of step or num has to be set")
            raise ValueError("Either one of step or num has to be set")

        if step:
            num = int(abs(start - stop) / step)
        if ramprate:
            delay = abs(start - stop) / (ramprate * num)

        if not isinstance(parameter, Sequence):
            self.parameter = [parameter]
        else:
            self.parameter = parameter

        if spacing == "log":
            self.values = np.logspace(np.log10(start), np.log10(stop), num)
        elif spacing == "lin":
            self.values = np.linspace(start, stop, num)
        else:
            logger.error("Invalid spacing type. Use 'lin' or 'log'.")
            raise ValueError("Invalid spacing type. Use 'lin' or 'log'.")

        self.start = start
        self.stop = stop
        self.num = num
        self.delay = delay
        self.start_delay = start_delay


class CircularSweep:
    def __init__(
        self,
        parameter: Parameter | Sequence[Parameter],
        start: int | float,
        stop: int | float,
        num: int | float = 0.0,
        step: int | float = 0.0,
        delay: float = 0.0,
        start_delay: float = 0.0,
        ramprate: float = 0.0,
        repetitions: int = 1,
    ) -> None:
        """
        Define a sweep from start to stop and back, optionally repeated.

        Args:
            parameter (Parameter | Sequence[Parameter]): QCoDeS parameter or
                parameters that share the same setpoints.
            start (int | float): First setpoint of the outward leg.
            stop (int | float): Last setpoint of the outward leg.
            num (int | float): Points in each direction. Ignored when *step* is
                nonzero.
            step (int | float): Step-size request converted to a point count.
            delay (float): Nonzero dwell time in seconds after each setpoint.
            start_delay (float): Delay in seconds after moving to the first
                setpoint. Defaults to 0.
            ramprate (float): If nonzero, derive *delay* from this
                units-per-second rate.
            repetitions (int): Number of complete outward-and-return cycles.
                Defaults to 1.

        Raises:
            ValueError: If neither *num* nor *step* is provided, or if neither
                *delay* nor *ramprate* is nonzero.

        Notes:
            Both endpoints occur twice in a cycle because each direction is an
            inclusive ``linspace``.

        """
        if not (step or num):
            logger.error("Either one of step or num has to be set")
            raise ValueError("Either one of step or num has to be set")
        if not (delay or ramprate):
            logger.error("Either one of delay or ramprate has to be set")
            raise ValueError("Either one of delay or ramprate has to be set")

        if step:
            num = int(abs(start - stop) / step)
        if ramprate:
            delay = abs(start - stop) / (ramprate * num)

        self.parameter = np.atleast_1d(parameter)
        self.values = np.tile(
            np.concatenate(
                [np.linspace(start, stop, num), np.linspace(start, stop, num)[::-1]]
            ),
            reps=repetitions,
        )
        self.start = start
        self.stop = stop
        self.num = num
        self.delay = delay
        self.start_delay = start_delay


class PointSweep:
    def __init__(
        self,
        parameter: Parameter | Sequence[Parameter],
        points: Sequence[int | float],
        delay: float = 0.0,
        start_delay: float = 0.0,
    ) -> None:
        """
        Define a sweep over an explicit sequence of setpoints.

        Args:
            parameter (Parameter | Sequence[Parameter]): QCoDeS parameter or
                parameters that share the same setpoints.
            points (Sequence[int | float]): Setpoints in traversal order.
            delay (float): Dwell time in seconds after each setpoint. Defaults
                to 0.
            start_delay (float): Delay in seconds after moving to the first
                setpoint. Defaults to 0.

        Raises:
            ValueError: If points is not a non-empty sequence of finite real
                numbers.

        """

        # NumPy arrays are not runtime Sequences; callers must pass a list.
        if (
            not isinstance(points, Sequence)
            or isinstance(points, (str, bytes))
            or len(points) == 0
            or not all(
                isinstance(point, Real)
                and not isinstance(point, bool)
                and isfinite(point)
                for point in points
            )
        ):
            logger.error("Points must be a sequence of finite real numbers")
            raise ValueError("Points must be a sequence of finite real numbers")

        if not isinstance(parameter, Sequence):
            self.parameter = [parameter]
        else:
            self.parameter = parameter

        self.values = np.array(points)

        self.start = points[0]
        self.stop = points[-1]
        self.num = len(points)
        self.delay = delay
        self.start_delay = start_delay


class SegmentedSweep:
    def __init__(
        self,
        parameter: Parameter,
        start: int | float,
        stop: int | float,
        num: int | float,
        center: int | float,
        center_width: int | float,
        factor: float = 10.0,
    ) -> None:
        """
        Define a three-part linear sweep with finer sampling near a center.

        Args:
            parameter (Parameter): QCoDeS parameter to sweep.
            start (int | float): First setpoint.
            stop (int | float): Final setpoint.
            num (int | float): Target point count distributed across the three
                regions before per-region counts are rounded down.
            center (int | float): Center of the finely sampled region.
            center_width (int | float): Width of the finely sampled region.
            factor (float): Sampling-density multiplier in the center region.
                Defaults to 10.

        """

        n_p = num / (stop - start + center_width * (factor - 1))

        self.segments = [
            {
                "start": start,
                "stop": center - center_width / 2,
                "points": n_p * (center - center_width / 2 - start),
            },
            {
                "start": center - center_width / 2,
                "stop": center + center_width / 2,
                "points": factor * n_p * center_width,
            },
            {
                "start": center + center_width / 2,
                "stop": stop,
                "points": n_p * (stop - center - center_width / 2),
            },
        ]

        self.parameter = np.atleast_1d(parameter)
        self.values = np.array([])
        for segment in self.segments:
            seg_values = np.linspace(
                segment["start"], segment["stop"], int(segment["points"])
            )
            self.values = np.concatenate((self.values, seg_values))
        self.start = start
        self.stop = stop
        self.num = num

        # Measurement expects every sweep to expose these attributes.
        self.delay = 0.0
        self.start_delay = 0.0


def _sweep_param(sweep: Sweep = None, override: list = None):
    """
    Set one parameter at each point in a sweep.

    Args:
        sweep (Sweep): Sweep definition. Ignored when *override* is provided.
        override (list): Optional ``[parameter, values, delay]`` tuple used by
            parallel sweeps.
    """
    if override:
        param, values, delay = override
    else:
        param = sweep.parameter
        values = sweep.values
        delay = sweep.delay

    for value in values:
        param(value)
        sleep(delay)


def _sweep_parameters(sweep: Sweep) -> list:
    """
    Return a sweep's parameters as a flat list.

    Sweep classes store parameters as either sequences or NumPy arrays. This
    helper normalizes both forms for callers.

    Args:
        sweep (Sweep): Sweep whose parameters are needed.

    Returns:
        list: Parameters in the sweep.

    """
    parameter = sweep.parameter
    if isinstance(parameter, (Sequence, np.ndarray)) and not isinstance(
        parameter, (str, bytes)
    ):
        return list(parameter)
    return [parameter]


def sweeper(sweeps: Sweep | Sequence[Sweep], parallel: bool = False):
    """
    Move parameters through one or more sweeps without recording data.

    Args:
        sweeps (Sweep | Sequence[Sweep]): Sweep or sweeps to execute.
        parallel (bool): Start each parameter sweep on its own thread. The
            function does not wait for those threads to finish. Defaults to
            False.

    """
    sweep_list = sweeps if isinstance(sweeps, Sequence) else [sweeps]

    for sweep in sweep_list:
        for parameter in _sweep_parameters(sweep):
            override = [parameter, sweep.values, sweep.delay]
            if parallel:
                # Pass the parameter explicitly so the thread does not capture
                # the loop variable by reference.
                Thread(target=_sweep_param, kwargs={"override": override}).start()
            else:
                _sweep_param(override=override)


def rampdown(parameters: Parameter | Sequence[Parameter], ramprate: float = None):
    """
    Move one or more parameters from their current values to zero.

    Args:
        parameters (Parameter | Sequence[Parameter]): QCoDeS parameters to
            ramp down.
        ramprate (float): Optional maximum units-per-second rate. When omitted,
            use a 0.01-second dwell. Each ramp contains 500 setpoints.

    """
    delay = 0.0
    if ramprate:
        if isinstance(parameters, Sequence):
            for parameter in parameters:
                new_delay = abs(parameter()) / (ramprate * 500)
                if new_delay > delay:
                    delay = new_delay
        else:
            delay = abs(parameters()) / (ramprate * 500)
    else:
        delay = 1e-2

    if isinstance(parameters, Sequence):
        sweeper(
            [
                Sweep(parameter, parameter(), 0.0, num=500, delay=delay)
                for parameter in parameters
            ]
        )
        return
    sweeper(Sweep(parameters, parameters(), 0.0, num=500, delay=delay))


def _stepper(
    dataset,
    data_location: str,
    depth: float,
    sweeps: Sequence,
    independents: Sequence,
    dependents: Sequence,
    sweep_cache: Sequence,
    bar,
    save_interval: float = 0.1,
    interrupt: Callable = lambda: False,
    parallel_sweep: bool = False,
    memory_store: Any | None = None,
    disk_store: Any | None = None,
    nc_snapshot_path: Optional[str] = None,
    verbose: bool = False,
    buffered_sweep: Optional[Any] = None,
    sweep_index_cache: Optional[list] = None,
):
    """
    Run nested sweeps and checkpoint acquired data.

    Args:
        dataset: Preallocated xarray dataset to fill.
        data_location: Fallback checkpoint path.
        depth: Current depth in the nested slow sweeps.
        sweeps: Slow sweeps in outer-to-inner order.
        independents: Swept parameters used as dataset coordinates.
        dependents: QCoDeS parameters read at each point.
        sweep_cache: Current coordinate value for each independent.
        bar: Progress bar updated after each point.
        save_interval: Minimum seconds between checkpoints.
        interrupt: Callback that requests an early stop when true.
        parallel_sweep: Set parameters on separate threads.
        memory_store: In-memory Zarr store used for live data.
        disk_store: Zarr store used for recovery checkpoints.
        nc_snapshot_path: Optional path for best-effort netCDF snapshots.
        verbose: Log checkpoint operations.
        buffered_sweep: Optional tree of hardware-buffered sweeps.
        sweep_index_cache: Current positional index for each independent.
    """
    global last_save
    depth = int(depth)

    if sweep_index_cache is None:
        sweep_index_cache = [None] * len(independents)

    # Measurement.run supplies buffered trees separately. Keep support for
    # legacy direct callers that pass one in `sweeps`.
    has_duplicate_sweep_values = any(
        len(sw.values) != len(set(sw.values)) for sw in sweeps
    )

    if buffered_sweep is None and len(sweeps) == 1 and not hasattr(sweeps[0], "values"):
        buffered_sweep = sweeps[0]
        sweeps = []
        depth = 1

    def _assign(arr, value, slow_indexers, slow_position_indexers):
        if has_duplicate_sweep_values:
            position = {
                dim: slow_position_indexers[dim]
                for dim in arr.dims
                if dim in slow_position_indexers
            }
            arr[position] = value
        else:
            arr.loc[slow_indexers] = value

    # Reuse opened arrays because parsing their metadata is relatively expensive.
    memory_arrays: dict = {}

    def _persist_memory(position=None):
        """
        Flush acquired points to the in-memory store the live server reads.

        When the acquired position is known, only rows changed since the last
        flush are written. Otherwise the whole store is replaced.

        Args:
            position (Any): Coordinate values of the point just acquired, or
                ``None`` to rewrite the whole store.

        """
        if memory_store is None:
            logger.warning(
                "[_stepper._persist_memory] No memory store provided, skipping in-memory persistence"
            )
            return

        key = id(memory_store)
        region = _memory_region(dataset, position, _MEMORY_FLUSHED_ROWS.get(key, 0))

        if region is not None:
            low, high, outer = region
            try:
                rows = {
                    name: variable.values[low:high]
                    for name, variable in dataset.data_vars.items()
                }
                sync(_write_rows(memory_store, memory_arrays, rows, low, high))
            except Exception as exc:
                # An unseeded store has no group to open. Fall through to the
                # full write, which creates one.
                logger.debug(
                    "[_stepper._persist_memory] Region write unavailable (%s); "
                    "rewriting the whole store",
                    exc,
                )
            else:
                # The current row is still being filled, so it stays dirty.
                _MEMORY_FLUSHED_ROWS[key] = max(0, high - 1)
                if verbose:
                    logger.info(
                        "[_stepper._persist_memory] Wrote rows %d:%d of '%s' to "
                        "the in-memory store",
                        low,
                        high,
                        outer,
                    )
                return

        memory_arrays.clear()
        dataset.to_zarr(store=memory_store, mode="w", zarr_format=ZARR_FORMAT)
        _MEMORY_FLUSHED_ROWS[key] = 0
        if verbose:
            logger.info("[_stepper._persist_memory] Saved dataset to in-memory store")

    def _persist_disk():
        for attempt in range(1, DISK_PERSIST_ATTEMPTS + 1):
            try:
                if disk_store is not None:
                    if memory_store is not None:
                        written = _copy_changed_keys(
                            memory_store,
                            disk_store,
                            _store_cache_key(disk_store),
                        )
                        if verbose:
                            logger.info(
                                "[_stepper._persist_disk] Copied %d changed key(s) "
                                "of %d from the in-memory store to the disk store",
                                written,
                                len(memory_store),
                            )
                    else:
                        dataset.to_zarr(
                            store=disk_store, mode="a", zarr_format=ZARR_FORMAT
                        )
                        if verbose:
                            logger.info(
                                "[_stepper._persist_disk] Saved dataset directly to disk store"
                            )
                else:
                    dataset.to_zarr(
                        store=data_location, mode="w", zarr_format=ZARR_FORMAT
                    )
                    if verbose:
                        logger.info(
                            f"[_stepper._persist_disk] Saved dataset to {data_location}"
                        )
                break
            except PermissionError as exc:
                if attempt == DISK_PERSIST_ATTEMPTS:
                    raise

                delay = DISK_PERSIST_RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "[_stepper._persist_disk] Disk store is temporarily locked "
                    "(attempt %d/%d); retrying in %.1f s: %s",
                    attempt,
                    DISK_PERSIST_ATTEMPTS,
                    delay,
                    exc,
                )
                sleep(delay)

        if nc_snapshot_path:
            try:
                tmp_nc = f"{nc_snapshot_path}.tmp"
                dataset.to_netcdf(tmp_nc, mode="w")
                os.replace(tmp_nc, nc_snapshot_path)
                if verbose:
                    logger.info(
                        "[_stepper._persist_disk] Wrote netCDF snapshot to %s",
                        nc_snapshot_path,
                    )
            except Exception as exc:
                logger.warning(
                    "[_stepper._persist_disk] Failed netCDF snapshot to %s: %s",
                    nc_snapshot_path,
                    exc,
                )

    # Keep the last position available for the final checkpoint at this depth.
    last_position: dict = {}

    def _checkpoint(position=None):
        """Persist live state without dropping points during a file lock."""
        _persist_memory(position)
        try:
            _persist_disk()
        except PermissionError as exc:
            logger.error(
                "[_stepper._checkpoint] Disk checkpoint remains locked after %d "
                "attempts; continuing the sweep with the complete in-memory "
                "dataset. A later checkpoint will retry: %s",
                DISK_PERSIST_ATTEMPTS,
                exc,
            )

    def _run_buffered_block(slow_indexers, slow_position_indexers):
        """
        Run, fetch, and store one complete hardware-buffered block.

        Args:
            slow_indexers (dict[str, Any]): Coordinate selections for the
                surrounding slow sweeps.
            slow_position_indexers (dict[str, int]): Positional selections for
                slow dimensions whose coordinate labels are repeated.

        Returns:
            dict[Any, dict[str, Any]]: A mapping from buffered dependent
                parameters to their fetched result metadata and arrays.

        """
        points, estimated_duration = _buffered_sweep_progress_info(buffered_sweep)
        progress = _BufferedProgress(bar, points, estimated_duration)
        progress.start()

        try:
            # Only estimated progress updates run in the background.
            toplevel = _arm_instruments(buffered_sweep)
            toplevel.run_sweep()

            results_state = {}
            _fetch_results(buffered_sweep, state=results_state)
            buffered_results_tree = results_state.get("results_tree", {})

            # Selecting the slow dimensions assigns the complete buffered block.
            for dep, entry in buffered_results_tree.items():
                arr = dataset.data_vars[f"{dep.name}"]
                _assign(
                    arr,
                    entry["result"],
                    slow_indexers,
                    slow_position_indexers,
                )
        except BaseException:
            progress.stop()
            raise

        progress.complete()
        return buffered_results_tree

    try:
        if len(sweeps) == 0:
            slow_indexers = {}
            slow_position_indexers = {}

            # Buffered dependents are handled separately by `_run_buffered_block`.
            for dependent in dependents:
                arr = dataset.data_vars[f"{dependent.name}"]
                _assign(
                    arr,
                    dependent(),
                    slow_indexers,
                    slow_position_indexers,
                )

            if buffered_sweep is not None:
                _run_buffered_block(slow_indexers, slow_position_indexers)
            else:
                bar.update(1)

            if last_save == 0 or (time() - last_save > save_interval):
                last_save = time()
                _checkpoint()

            _checkpoint()
            return dataset

        sweep = sweeps[len(sweeps) - depth]

        for idx, sweep_point in enumerate(sweep.values):
            if idx == 0:
                sleep(sweep.start_delay)

            if interrupt():
                logger.info("Interrupt received from measurement parameters")
                raise InterruptedError("Interrupt received from measurement parameters")

            else:
                for param in sweep.parameter:
                    if parallel_sweep:
                        Thread(target=lambda: param(sweep_point)).start()
                    else:
                        param(sweep_point)
                    if param in independents:
                        cache_idx = independents.index(param)
                        sweep_cache[cache_idx] = sweep_point
                        sweep_index_cache[cache_idx] = idx
                sleep(sweep.delay)

            if depth > 1:
                _stepper(
                    dataset=dataset,
                    data_location=data_location,
                    depth=depth - 1,
                    sweeps=sweeps,
                    independents=independents,
                    dependents=dependents,
                    sweep_cache=sweep_cache,
                    bar=bar,
                    save_interval=save_interval,
                    interrupt=interrupt,
                    parallel_sweep=parallel_sweep,
                    memory_store=memory_store,
                    disk_store=disk_store,
                    nc_snapshot_path=nc_snapshot_path,
                    verbose=verbose,
                    buffered_sweep=buffered_sweep,
                    sweep_index_cache=sweep_index_cache,
                )

            elif depth == 1:
                # Build indexers for slow dimensions; the buffered sweep owns
                # its own dimensions.
                slow_indexers = {}
                slow_position_indexers = {}

                for sw in sweeps:
                    for p in sw.parameter:
                        try:
                            cache_idx = independents.index(p)
                        except ValueError:
                            logger.error(
                                f"Independent parameter {p.name} not found in independents list, skipping"
                            )
                            continue
                        slow_indexers[p.name] = sweep_cache[cache_idx]
                        slow_position_indexers[p.name] = sweep_index_cache[cache_idx]

                # Buffered dependents are handled separately below.
                for dependent in dependents:
                    arr = dataset.data_vars[f"{dependent.name}"]
                    _assign(
                        arr,
                        dependent(),
                        slow_indexers,
                        slow_position_indexers,
                    )

                if buffered_sweep is not None:
                    # Buffered instruments must be re-armed at each slow point.
                    _run_buffered_block(slow_indexers, slow_position_indexers)
                else:
                    bar.update(1)

                last_position = slow_indexers

                if last_save == 0 or (time() - last_save > save_interval):
                    last_save = time()
                    _checkpoint(slow_indexers)
    except (Exception, KeyboardInterrupt):
        # Preserve acquired data, then re-raise the original failure.
        try:
            _persist_memory()
        except Exception:
            logger.exception(
                "[_stepper] Failed to preserve in-memory state while handling "
                "a measurement error"
            )
        raise

    _checkpoint(last_position or None)
    return dataset
