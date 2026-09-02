import os
from threading import Event, Thread
from time import monotonic, sleep, time
from typing import Any, Callable, Optional, Sequence

import numpy as np
import zarr
from qcodes.parameters import Parameter

from qcutils.buffered.sweep import (
    arm_instruments,
    _buffered_sweep_progress_info,
    fetch_results,
)
from qcutils.logger import get_logger

logger = get_logger(__name__)
last_save = 0

DISK_PERSIST_ATTEMPTS = 6
DISK_PERSIST_RETRY_DELAY = 0.2


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
        refresh_interval = min(
            0.1, max(0.01, self.estimated_duration / self.points)
        )

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
        """Sweep class to define a parameter sweep

        Args:
            parameter: qcodes parameter(s) to sweep, if list then all of them are swept according to the same sweep conditions
            start: start point
            stop: stop point
            step: stepsize
            num: number of points
            delay: delay / dwell time between points
            ramprate: ramp rate
            spacing: spacing of the sweep points ("lin" or "log")
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
        """Sweep class to define a circular parameter sweep, where the parameter goes up and comes down

        Args:
            parameter: qcodes parameter(s) to sweep, if list then all of them are swept according to the same sweep conditions
            start: start point
            stop: stop point
            step: stepsize
            num: number of points
            delay: delay / dwell time between points
            ramprate: ramp rate
            repetitions: number of times to repeat the sweep
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
        """Sweep class to define a resonator parameter sweep, where the parameter goes from start to stop with finer steps around center

        Args:
            parameter: qcodes parameter to sweep
            start: start point
            stop: stop point
            num: number of points
            center: center point of the resonator
            center_width: width around center with finer steps
            factor: Scaling factor for the center width region

        """

        # build the three segments of the sweep
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

        # need to be accsessed in measurement (only dummy)
        self.delay = 0.0
        self.start_delay = 0.0


def _sweep_param(sweep: Sweep = None, override: list = None):
    """Does a single sweep

    Args:
     sweep
     override: List of parameter, values and delay to manualy do a sweep, and override Sweep
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


def sweeper(sweeps: Sweep | Sequence[Sweep], parallel: bool = False):
    """Sweeps through anything without making a measurement

    Args:
        sweeps: Sweep, or list of sweeps
        parallel: Boolean to enable parallel sweeps
    """
    if parallel:
        if isinstance(sweeps, Sequence):
            for sweep in sweeps:
                if isinstance(sweep.parameter, Sequence):
                    for parameter in sweep.parameter:
                        Thread(
                            target=lambda: _sweep_param(
                                override=[parameter, sweep.values, sweep.delay]
                            )
                        ).start()
                else:
                    Thread(target=lambda: _sweep_param(sweep)).start()

        else:
            if isinstance(sweeps.parameter, Sequence):
                for parameter in sweeps.parameter:
                    Thread(
                        target=lambda: _sweep_param(
                            override=[parameter, sweeps.values, sweeps.delay]
                        )
                    ).start()
            else:
                _sweep_param(sweeps)

    else:
        if isinstance(sweeps, Sequence):
            for sweep in sweeps:
                if isinstance(sweep.parameter, Sequence):
                    for parameter in sweep.parameter:
                        _sweep_param(override=[parameter, sweep.values, sweep.delay])
                else:
                    _sweep_param(sweep)
        else:
            if isinstance(sweeps.parameter, Sequence):
                for parameter in sweeps.parameter:
                    _sweep_param(override=[parameter, sweeps.values, sweeps.delay])
            else:
                _sweep_param(sweeps)


def rampdown(parameters: Parameter | Sequence[Parameter], ramprate: float = None):
    """Ramp random parameters to zero

    Arg:
        parameters
        ramprate : Ramprate for ramped instruments
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


def stepper(
    dataset,
    data_location: str,
    depth: int | float,
    sweeps: Sequence,
    independents: Sequence,
    dependents: Sequence,
    sweep_cache: Sequence,
    bar,
    save_interval: float = 0.1,
    interrupt: Callable = lambda: False,
    parallel_sweep: bool = False,
    memory_store: Optional[Any] = None,
    disk_store: Optional[Any] = None,
    nc_snapshot_path: Optional[str] = None,
    verbose: bool = False,
    buffered_sweep: Optional[Any] = None,
):
    """
    Recursive stepper function for generating the for loops required to sweep measurements

    Args:
        dataset: xarray dataset to store the data
        data_location: Location to store the data
        depth: Depth of the nested for loops to be emulated
        sweeps: List of sweeps for each for loop (slow sweeps only; buffered sweeps handled separately)
        independents: List of independent parameters that are to be plotted, with the measured parameters
        dependents: List of dependent parameters (QCoDeS parameters)
        sweep_cache: Cache of the last sweep parameters (one slot per entry in `independents`)
        save_interval: Time interval to save the data
        interrupt: Boolean to enable custom stops
        parallel_sweep: Boolean to enable parallel parameter sweeps
        memory_store: Optional zarr store that keeps the live dataset in memory
        disk_store: Optional zarr store used for persisting to disk
        nc_snapshot_path: Optional netcdf path for best-effort in-run snapshots
        verbose: Enables verbose logging of persistence actions when True
        buffered_sweep: Optional buffered sweep tree describing fast, instrument-internal sweeps
    """
    global last_save
    depth = int(depth)

    if buffered_sweep is None and len(sweeps) == 1 and not hasattr(sweeps[0], "values"):
        buffered_sweep = sweeps[0]
        sweeps = []
        depth = 1

    def _persist_memory():
        if memory_store is None:
            logger.warning(
                "[stepper._persist_memory] No memory store provided, skipping in-memory persistence"
            )
            return
        dataset.to_zarr(store=memory_store, mode="w")
        if verbose:
            logger.info("[stepper._persist_memory] Saved dataset to in-memory store")

    def _persist_disk():
        for attempt in range(1, DISK_PERSIST_ATTEMPTS + 1):
            try:
                if disk_store is not None:
                    if memory_store is not None:
                        zarr.copy_store(memory_store, disk_store, if_exists="replace")
                        if verbose:
                            logger.info(
                                "[stepper._persist_disk] Copied in-memory store to disk store"
                            )
                    else:
                        dataset.to_zarr(store=disk_store, mode="a")
                        if verbose:
                            logger.info(
                                "[stepper._persist_disk] Saved dataset directly to disk store"
                            )
                else:
                    dataset.to_zarr(store=data_location, mode="w")
                    if verbose:
                        logger.info(
                            f"[stepper._persist_disk] Saved dataset to {data_location}"
                        )
                break
            except PermissionError as exc:
                if attempt == DISK_PERSIST_ATTEMPTS:
                    raise

                delay = DISK_PERSIST_RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "[stepper._persist_disk] Disk store is temporarily locked "
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
                        "[stepper._persist_disk] Wrote netCDF snapshot to %s",
                        nc_snapshot_path,
                    )
            except Exception as exc:
                logger.warning(
                    "[stepper._persist_disk] Failed netCDF snapshot to %s: %s",
                    nc_snapshot_path,
                    exc,
                )

    def _checkpoint():
        """Persist live state without letting a transient file lock skip sweep points."""
        _persist_memory()
        try:
            _persist_disk()
        except PermissionError as exc:
            logger.error(
                "[stepper._checkpoint] Disk checkpoint remains locked after %d "
                "attempts; continuing the sweep with the complete in-memory "
                "dataset. A later checkpoint will retry: %s",
                DISK_PERSIST_ATTEMPTS,
                exc,
            )

    def _run_buffered_block(slow_indexers):
        """
        Run, fetch, and store one complete hardware-buffered block.

        Args:
            slow_indexers (dict[str, Any]): Coordinate selections for the
                surrounding slow sweeps.

        Returns:
            dict[Any, dict[str, Any]]: A mapping from buffered dependent
                parameters to their fetched result metadata and arrays.

        """
        points, estimated_duration = _buffered_sweep_progress_info(buffered_sweep)
        progress = _BufferedProgress(bar, points, estimated_duration)
        progress.start()

        try:
            # Instrument access stays on the caller thread. Only the estimated
            # tqdm updates happen on _BufferedProgress's background thread.
            toplevel = arm_instruments(buffered_sweep)
            toplevel.run_sweep()

            results_state = {}
            fetch_results(buffered_sweep, state=results_state)
            buffered_results_tree = results_state.get("results_tree", {})

            # Each dependent DataArray has [slow dims..., buffered dims...].
            # Selecting only slow dimensions assigns the complete fast block.
            for dep, entry in buffered_results_tree.items():
                arr = dataset.data_vars[f"{dep.name}"]
                arr.loc[slow_indexers] = entry["result"]
        except BaseException:
            progress.stop()
            raise

        progress.complete()
        return buffered_results_tree

    try:
        if len(sweeps) == 0:
            slow_indexers = {}

            buffered_results_tree = {}
            buffered_dependents = set()

            for dependent in dependents:
                if dependent in buffered_dependents:
                    logger.error(
                        f"Dependent {dependent.name} is provided by buffered sweep; skipping direct read to avoid double-reading."
                    )
                    continue

                arr = dataset.data_vars[f"{dependent.name}"]
                arr.loc[slow_indexers] = dependent()

            if buffered_sweep is not None:
                buffered_results_tree = _run_buffered_block(slow_indexers)
                buffered_dependents = set(buffered_results_tree.keys())
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
                        sweep_cache[independents.index(param)] = sweep_point
                sleep(sweep.delay)

            if depth > 1:
                stepper(
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
                )

            elif depth == 1:
                # At the deepest level: evaluate dependents and (optionally)
                # run a buffered sweep tree.

                # Build indexers for the "slow" dimensions from sweeps.
                # We deliberately ignore any buffered dims here; they are
                # spanned by the buffered sweep itself.
                slow_indexers = {}
                for sw in sweeps:
                    for p in sw.parameter:
                        try:
                            idx = independents.index(p)
                        except ValueError:
                            logger.error(
                                f"Independent parameter {p.name} not found in independents list, skipping"
                            )
                            continue
                        slow_indexers[p.name] = sweep_cache[idx]

                buffered_results_tree = {}
                buffered_dependents = set()

                for dependent in dependents:
                    if dependent in buffered_dependents:
                        # This dependent is provided by buffered instruments;
                        # do not read it directly to avoid double-reading.
                        logger.error(
                            f"Dependent {dependent.name} is provided by buffered sweep; skipping direct read to avoid double-reading."
                        )
                        continue

                    arr = dataset.data_vars[f"{dependent.name}"]
                    arr.loc[slow_indexers] = dependent()

                if buffered_sweep is not None:
                    # Re-arm instruments for this buffered tree at every
                    # slow-step: instruments can only be read once after a sweep.
                    buffered_results_tree = _run_buffered_block(slow_indexers)
                    buffered_dependents = set(buffered_results_tree.keys())
                else:
                    bar.update(1)

                if last_save == 0 or (time() - last_save > save_interval):
                    last_save = time()
                    # NOTE: Writing dataset to MemoryStore as well as DiskStore
                    _checkpoint()
    except (Exception, KeyboardInterrupt):
        # Preserve the latest acquired data, then let the caller handle the interruption/failure.
        try:
            _persist_memory()
        except Exception:
            logger.exception(
                "[stepper] Failed to preserve in-memory state while handling "
                "a measurement error"
            )
        raise

    _checkpoint()
    return dataset
