from qcodes.parameters import Parameter

import numpy as np

from typing import Union, Sequence
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
