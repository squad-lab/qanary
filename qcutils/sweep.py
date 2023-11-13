# keeping qcutils with the sweeps as is for now
from qcodes.dataset import Measurement
from qcodes.parameters import Parameter
from qcodes.dataset.experiment_container import Experiment

import numpy as np

from typing import Union, Sequence, Callable
from threading import Thread
from time import sleep


class Sweep:
    def __init__(
        self,
        parameter: Union[Parameter, Sequence[Parameter]],
        start: Union[int, float],
        stop: Union[int, float],
        num: Union[int, float] = 0.0,
        step: Union[int, float] = 0.0,
        delay: float = 0.0,
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
            raise ValueError("Either one of step or num has to be set")
        if not (delay or ramprate):
            raise ValueError("Either one of delay or ramprate has to be set")

        if step:
            num = int(abs(start - stop) / step)
        if ramprate:
            delay = abs(start - stop) / (ramprate * num)

        self.parameter = parameter
        self.values = np.linspace(start, stop, num)
        self.delay = delay


class TimeSweep:
    def __init__(
        self,
        parameter: Parameter,
        duration: Union[int, float],
        step: Union[int, float],
    ) -> None:
        """Sweep class to define a parameter sweep

        Args:
            parameter: qcodes parameter(s) of clock e.g. clock.time()
            duration: overall length of the time sweep
            step: delay / dwell time between points
        """

        num = int(abs(duration) / step)

        self.parameter = parameter
        self.values = np.linspace(0, duration, num)


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
        """
        if not (step or num):
            raise ValueError("Either one of step or num has to be set")
        if not (delay or ramprate):
            raise ValueError("Either one of delay or ramprate has to be set")

        if step:
            num = int(abs(start - stop) / step)
        if ramprate:
            delay = abs(start - stop) / (ramprate * num)

        self.parameter = parameter
        self.values = np.append(
            np.linspace(start, stop, num), np.linspace(start, stop, num)[::-1]
        )
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


def sweeper(sweeps: Union[Sweep, Sequence[Sweep]]):
    """Sweeps through anything without making a measurement

    Args:
        sweeps
    """
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
            Thread(target=lambda: _sweep_param(sweeps)).start()


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
                new_delay = abs(parameter()) / (ramprate * 100)
                if new_delay > delay:
                    delay = new_delay
        else:
            delay = abs(parameters()) / (ramprate * 100)
    else:
        delay = 1e-2

    if isinstance(parameters, Sequence):
        sweeper(
            [
                Sweep(parameter, parameter(), 0.0, num=100, delay=delay)
                for parameter in parameters
            ]
        )
        return
    sweeper(Sweep(parameters, parameters(), 0.0, num=100, delay=delay))


def _stepper(
    depth: Union[int, float],
    sweep_list: Sequence,
    plot_independents: Sequence,
    dependents: Sequence,
    sweep_cache: Sequence,
    datasaver,
    interrupt: bool,
    offset: float,
):
    """Recursive stepper function for generating the for loops required to sweep measurements

    Args:
        depth: Depth of the nested for loops to be emulated
        sweep_list: List of sweeps for each for loop
        plot_independents: List of independent parameters that are to be plotted, with the measured parameters
        dependents: List of dependent parameters
        sweep_cache: Cache of the last sweep parameters
        datasaver: qcodes datasaver
        interrupt: Boolean to enable custom stops
    """
    sweep = sweep_list[len(sweep_list) - depth]
    sweep_step = sweep.values[1] - sweep.values[0]
    if isinstance(sweep, TimeSweep):
        timesweep_start = sweep.parameter()

    for sweep_point in sweep.values:
        if interrupt:
            raise InterruptedError("Interrupt recieved from measurement parameters")

        if isinstance(sweep, TimeSweep):
            while True:
                if timesweep_start + sweep_point < sweep.parameter():
                    break
            sweep_cache[plot_independents.index(sweep.parameter)] = (
                sweep.parameter() - offset
            )

        else:
            if isinstance(sweep.parameter, Sequence):
                for param in sweep.parameter:
                    param(sweep_point)
                    while True:
                        if abs(param() - sweep_point) < 0.01 * abs(sweep_step):
                            break
                    if param in plot_independents:
                        sweep_cache[plot_independents.index(param)] = (
                            sweep_point - offset
                        )

            else:
                sweep.parameter(sweep_point)
                while True:
                    if abs(sweep.parameter() - sweep_point) < 0.01 * abs(sweep_step):
                        break
                sweep_cache[plot_independents.index(sweep.parameter)] = (
                    sweep_point - offset
                )

            sleep(sweep.delay)

        if depth == 1:
            datasaver.add_result(
                *(
                    list(zip(plot_independents, sweep_cache))
                    + [(dependent, dependent()) for dependent in dependents]
                )
            )

        if depth > 1:
            _stepper(
                depth - 1,
                sweep_list,
                plot_independents,
                dependents,
                sweep_cache,
                datasaver,
                interrupt,
                offset,
            )
    return


def measure(
    sweeps: Union[Sweep, CircularSweep, TimeSweep, Sequence[Sweep]],
    parameters: dict,
    experiment: Experiment,
    measurement: str,
    write_period: float = 0.1,
    offset: float = 0.0,
    interrupt: Callable = None,
    rampdown_on_interrupt=False,
):
    """Measurement function to run a qcodes sweep based measurement

    Args:
        sweeps: Sequence of sweeps to be measured, can be a single element list or tuple too
        parameters: Dictionary of dependent parameters that depend on all independent parameters by default, otherwise depend on the independents key of the dictionary, example"
        meas_params = {
        "parameters": [dlockin.r, dlockin.p],
        "independents": [ch1, ch2]
        }
        experiment: qcodes Experiment to write to
        measurement: Name of the measurement
        write_period: Write period for the buffered writes to the database
        offset: Measurement offset on the independent variable
        interrupt: Boolean interrupting the measurement when true
        rampdown_on_interrupt: Rampdown on keyboard interrupt or errors

    Returns:
        dataset: qcodes dataset, if no interrupts have been issued
        1: If the measurement was interrupted
    """
    meas = Measurement(exp=experiment, name=measurement)
    meas.write_period = write_period

    independents = []
    if not isinstance(sweeps, Sequence):
        sweeps = [sweeps]

    for sweep in sweeps:
        if isinstance(sweep.parameter, Sequence):
            independents.append(sweep.parameter[0])
        else:
            independents.append(sweep.parameter)

    independents = list(np.asarray(independents).flatten())
    independents = list(set(independents))

    plot_independents = independents
    if "independents" in parameters:
        plot_independents = parameters["independents"]
    for independent in plot_independents:
        meas.register_parameter(independent)

    dependents = parameters["parameters"]
    for dependent in dependents:
        meas.register_parameter(dependent, setpoints=tuple(plot_independents))

    try:
        with meas.run() as datasaver:
            datasaver = _stepper(
                depth=len(sweeps),
                sweep_list=sweeps,
                plot_independents=plot_independents,
                dependents=dependents,
                sweep_cache=[0.0] * len(plot_independents),
                datasaver=datasaver,
                interrupt=interrupt,
                offset=offset,
            )
        dataset = datasaver.dataset
        return dataset

    except:
        if rampdown_on_interrupt:
            rampdown_sweeps = [
                Sweep(sweep.parameter, sweep.parameter(), 0.0, num=100, delay=1e-2)
                for sweep in sweeps
            ]
            sweeper(rampdown_sweeps)
            return 1
