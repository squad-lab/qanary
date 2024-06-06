from qcodes.dataset import Measurement
from qcodes.parameters import Parameter
from qcodes.dataset.experiment_container import Experiment

import numpy as np

from typing import Union, Sequence, Callable
from threading import Thread
from time import sleep
from tqdm import tqdm
from tabulate import tabulate

bar = None


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
    interrupt: Callable,
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

    # leveling the playing field to make every case a list
    if not isinstance(sweep.parameter, Sequence):
        sweep.parameter = [sweep.parameter]

    for idx, sweep_point in enumerate(sweep.values):
        if idx == 0:
            sleep(sweep.start_delay)

        if interrupt():
            raise InterruptedError("Interrupt recieved from measurement parameters")

        else:
            for param in sweep.parameter:
                # set the parameter to the setpoint
                param(sweep_point)
                if param in plot_independents:
                    sweep_cache[plot_independents.index(param)] = sweep_point

            sleep(sweep.delay)

        if depth > 1:
            _stepper(
                depth - 1,
                sweep_list,
                plot_independents,
                dependents,
                sweep_cache,
                datasaver,
                interrupt,
            )

        elif depth == 1:
            datasaver.add_result(
                *(
                    list(zip(plot_independents, sweep_cache))
                    + [(dependent, dependent()) for dependent in dependents]
                )
            )
            global bar
            bar.update(1)
    return datasaver


def measure(
    sweeps: Union[Sweep, CircularSweep, Sequence[Sweep]],
    parameters: dict,
    experiment: Experiment,
    measurement: str,
    interrupt: Callable = None,
    rampdown_on_interrupt=False,
):
    """Measurement function to run a qcodes sweep based measurement

    Args:
        sweeps: Sequence of sweeps to be measured, can be a single element list or tuple too
        parameters: Dictionary of dependent parameters that depend on all independent parameters by default, otherwise depend on the independents key of the dictionary, example"
        meas_params = {
        "parameters": [lockin.r, lockin.p],
        "independents": [ch1, ch2]
        }
        experiment: qcodes Experiment to write to
        measurement: Name of the measurement
        interrupt: Boolean interrupting the measurement when true
        rampdown_on_interrupt: Rampdown on keyboard interrupt or errors

    Returns:
        dataset: qcodes dataset, if no interrupts have been issued
        1: If the measurement was interrupted

    TBD:
        * Eliminate the need for defining independent sweep parameters, just take them from the Sweep object
    """
    meas = Measurement(exp=experiment, name=measurement)
    meas.write_period = 5e-3
    independents = []
    sweep_metadata = []
    sweep_metadata_headers = [
        "Independent(s)",
        "Range",
        "Number of Points",
        "Delay (s)",
    ]
    if not isinstance(sweeps, Sequence):
        sweeps = [sweeps]

    # allowing for multiple sweeps to be done one after the other in the same measurement
    for sweep in sweeps:
        sweep_metadata.append(
            [
                sweep.parameter,
                f"{min(sweep.values)}-{max(sweep.values)}",
                len(sweep.values),
                sweep.delay,
            ]
        )

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
        total_points = 1
        for sweep in sweeps:
            total_points *= len(sweep.values)

        with meas.run() as datasaver:
            print(
                tabulate(
                    sweep_metadata,
                    headers=sweep_metadata_headers,
                ),
                "\n",
            )
            global bar
            bar = tqdm(total=total_points, ascii="*ᗧⵔ●︎", desc="Measurement Progress")
            _stepper(
                depth=len(sweeps),
                sweep_list=sweeps,
                plot_independents=plot_independents,
                dependents=dependents,
                sweep_cache=[0.0] * len(plot_independents),
                datasaver=datasaver,
                interrupt=interrupt,
            )
            bar.close()
            return

    except KeyboardInterrupt:
        if rampdown_on_interrupt:
            rampdown_sweeps = [
                Sweep(sweep.parameter, sweep.parameter(), 0.0, num=100, delay=1e-2)
                for sweep in sweeps
            ]
            sweeper(rampdown_sweeps)
            return 1
