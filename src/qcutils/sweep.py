from threading import Thread
from time import sleep, time
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import zarr
from qcodes.parameters import Parameter

from qcutils.logger import get_logger
from qcutils.buffered.sweep import arm_instruments, fetch_results

logger = get_logger(__name__)
last_save = 0


class Sweep:
    def __init__(
        self,
        parameter: Union[Parameter, Sequence[Parameter]],
        start: Union[int, float],
        stop: Union[int, float],
        num: Union[int, float] = 0.0,
        step: Union[int, float] = 0.0,
        delay: float = 0.0,
        start_delay: float = 0.0,
        ramprate: float = 0.0,
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

        if not isinstance(parameter, Sequence):
            self.parameter = [parameter]
        else:
            self.parameter = parameter
        self.values = np.linspace(start, stop, num)
        self.start = start
        self.stop = stop
        self.num = num
        self.delay = delay
        self.start_delay = start_delay


class CircularSweep:
    def __init__(
        self,
        parameter: Union[Parameter, Sequence[Parameter]],
        start: Union[int, float],
        stop: Union[int, float],
        num: Union[int, float] = 0.0,
        step: Union[int, float] = 0.0,
        delay: float = 0.0,
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

        self.parameter = parameter
        self.values = np.append(
            np.tile(
                np.linspace(start, stop, num),
                np.linspace(start, stop, num)[::-1],
                reps=repetitions,
            )
        )
        self.start = start
        self.stop = stop
        self.num = num
        self.delay = delay


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


def sweeper(sweeps: Union[Sweep, Sequence[Sweep]], parallel: bool = False):
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


def rampdown(parameters: Union[Parameter, Sequence[Parameter]], ramprate: float = None):
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
    depth: Union[int, float],
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
            dataset.to_zarr(data_location, mode="a")
            if verbose:
                logger.info(f"[stepper._persist_disk] Saved dataset to {data_location}")

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
                toplevel = arm_instruments(buffered_sweep)
                toplevel.run_sweep()

                results_state = {}
                fetch_results(buffered_sweep, state=results_state)
                buffered_results_tree = results_state.get("results_tree", {})
                buffered_dependents = set(buffered_results_tree.keys())

            for dep, entry in buffered_results_tree.items():
                arr = dataset.data_vars[f"{dep.name}"]
                arr.loc[slow_indexers] = entry["result"]

            bar.update(1)

            if last_save == 0 or (time() - last_save > save_interval):
                last_save = time()
                _persist_memory()
                _persist_disk()

            _persist_memory()
            _persist_disk()
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
                    toplevel = arm_instruments(buffered_sweep)
                    toplevel.run_sweep()

                    # Fetch all buffered results
                    results_state = {}
                    fetch_results(buffered_sweep, state=results_state)
                    buffered_results_tree = results_state.get("results_tree", {})
                    buffered_dependents = set(buffered_results_tree.keys())

                # Write buffered results into the dataset.
                # For each buffered dependent, its DataArray has dimensions
                # [slow_dims..., fast_dims...].  By indexing only on slow
                # dims, we assign an entire block over the fast dims.
                for dep, entry in buffered_results_tree.items():
                    arr = dataset.data_vars[f"{dep.name}"]
                    arr.loc[slow_indexers] = entry["result"]

                bar.update(1)

                if last_save == 0 or (time() - last_save > save_interval):
                    last_save = time()
                    # NOTE: Writing dataset to MemoryStore as well as DiskStore
                    _persist_memory()
                    _persist_disk()
    except Exception as e:
        logger.exception(e, exc_info=True)
        _persist_memory()
        _persist_disk()

    _persist_memory()
    _persist_disk()
    return dataset
