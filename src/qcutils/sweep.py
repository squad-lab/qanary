import logging
from threading import Thread
from time import sleep, time
from typing import Callable, Sequence, Union

import numpy as np
from qcodes.parameters import Parameter

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
            logging.error("Either one of step or num has to be set")
            raise ValueError("Either one of step or num has to be set")
        if not (delay or ramprate):
            logging.error("Either one of delay or ramprate has to be set")
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
            logging.error("Either one of step or num has to be set")
            raise ValueError("Either one of step or num has to be set")
        if not (delay or ramprate):
            logging.error("Either one of delay or ramprate has to be set")
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


def sweeper(sweeps: Union[Sweep, Sequence[Sweep]], parallel=False):
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
):
    """Recursive stepper function for generating the for loops required to sweep measurements

    Args:
        dataset: xarray dataset to store the data
        data_location: Location to store the data
        depth: Depth of the nested for loops to be emulated
        sweeps: List of sweeps for each for loop
        independents: List of independent parameters that are to be plotted, with the measured parameters
        dependents: List of dependent parameters
        sweep_cache: Cache of the last sweep parameters
        save_interval: Time interval to save the data
        interrupt: Boolean to enable custom stops
        parallel_sweep: Boolean to enable parallel parameter sweeps
    """
    sweep = sweeps[len(sweeps) - depth]
    try:
        for idx, sweep_point in enumerate(sweep.values):
            if idx == 0:
                sleep(sweep.start_delay)

            if interrupt():
                logging.info("Interrupt recieved from measurement parameters")
                raise InterruptedError("Interrupt recieved from measurement parameters")

            else:
                for param in sweep.parameter:
                    # set the parameter to the setpoint
                    if parallel_sweep:
                        Thread(target=lambda: param(sweep_point)).start()
                    else:
                        param(sweep_point)
                    if param in independents:
                        sweep_cache[independents.index(param)] = sweep_point
                sleep(sweep.delay)

            if depth > 1:
                stepper(
                    dataset,
                    data_location,
                    depth - 1,
                    sweeps,
                    independents,
                    dependents,
                    sweep_cache,
                    bar,
                    save_interval,
                    interrupt,
                )

            elif depth == 1:
                for dependent in dependents:
                    independents_name = [param.name for param in independents]
                    arr = dataset.data_vars[f"{dependent.name}"]
                    arr.loc[dict(zip(independents_name, sweep_cache))] = dependent()

                bar.update(1)

                global last_save
                if last_save == 0 or (time() - last_save > save_interval):
                    last_save = time()
                    dataset.to_zarr(data_location, mode="a")
    except Exception as e:
        logging.exception(e)
        dataset.to_zarr(data_location, mode="a")

    dataset.to_zarr(data_location, mode="a")
    return dataset
