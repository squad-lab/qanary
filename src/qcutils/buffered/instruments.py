from time import sleep, time
from typing import Sequence, Union

import numpy as np
from qcodes.instrument import Instrument
from qcodes.parameters import Parameter

from qcutils.sweep import Sweep


class BufferedNodeBase:
    def __init__(self, inst: Instrument) -> None:
        """
        Base class for buffered sweep nodes. This class is a node of the buffered sweep tree. Can be used in three ways:
        1. As a node in a buffered sweep tree with a sweep and a dependent, where the device is both sweeping and measuring.
        2. As a node in a buffered sweep tree with a sweep and no dependent, where the device is only sweeping.
        3. As a node in a buffered sweep tree with no sweep and a dependent, where the device is only measuring.
        """
        self.buffered = True
        self.toplevel = False
        self._processed_sweeps = False
        self._processed_dependents = False
        self.core = inst

        self.endnode = True
        self.sweepnode = True

    def _process_sweeps(self, sweep: Sweep | Sequence[Sweep]):
        """
        Process sweeps

        Args:
            sweep (Sweep): The sweep object to be used in the buffered sweep tree.
        """
        if not isinstance(sweep, Sequence):
            self.sweeps = [sweep]
        else:
            self.sweeps = sweep

        instruments = [
            param.underlying_instrument for sw in self.sweeps for param in sw.parameter
        ]
        assert len(set(instruments)) == 1, (
            "All sweeps of the buffered node must be from the same instrument"
        )

        delays = [sw.delay for sw in self.sweeps]
        assert len(set(delays)) == 1, (
            "All sweeps of the buffered node must have the same delay"
        )

        assert len(self.sweeps) <= 2, "Maximum 2D sweep supported"

        self.delay = delays[0]
        self.num = 1
        for sw in self.sweeps:
            self.num *= sw.num
        self.num_tup = [sw.num for sw in self.sweeps]
        self.start_delay = [sw.start_delay for sw in self.sweeps]
        self.endnode = False
        self.dims = len(self.sweeps)

    def _process_dependents(
        self, dependent: Parameter | Sequence[Parameter] | str | Sequence[str]
    ):
        """
        Process dependents

        Args:
            dependent (Parameter | Sequence[Parameter]): The dependent object to be used in the buffered sweep tree.
        """
        if not isinstance(dependent, Sequence) or isinstance(dependent, (str, bytes)):
            self.dependents = [dependent]
        else:
            self.dependents = dependent

        self.sweepnode = False


class NodeMFLI(BufferedNodeBase):
    """
    MFLI node for a buffered sweep tree.

    - Uses the LabOne DAQ module in **hardware-trigger** mode (type=6).
    - Exact grid acquisition (grid/mode=2) with:
        rows := number of trigger events (outer dimension)
        cols := number of points recorded per trigger (inner dimension)

    Typical flow:
        register_dependent(...)  # config + subscribe + execute()
        ... run your QDAC sweep to generate triggers ...
        fetch()                  # wait for completion and read data
    """

    def __init__(self, inst: Instrument, *args, **kwargs) -> None:
        super().__init__(inst=inst, *args, **kwargs)

        self.core = self.core.core
        self.serial = self.core.serial
        self.daq = self.core.session.daq_server

        self.daq_module = self.daq.dataAcquisitionModule()
        self.daq_module.set("device", self.serial)
        self.daq_module.set("type", 6)
        self.daq_module.set("grid/mode", 2)

        self._subs: list[str] = []

    def register_dependent(
        self,
        dependent: str | Sequence[str],
        num: int | Sequence[int],
        delay: float,
        input_trigger: int = 1,
        *,
        edge: str = "rising",
        endless: bool = False,
        count: int = 1,
    ) -> None:
        """
        Configure DAQ for a hardware-triggered, exact-grid acquisition and start it.

        Args:
            dependent: LabOne node(s), e.g. "demods/0/sample.r" or list thereof.
            num: Number of points. For 2D, pass (rows, cols) where rows is the number
                 of trigger events (outer loop) and cols the points per trigger.
            delay: Step time (s) of the innermost sweep; DAQ duration ~ delay * cols.
            input_trigger: Which TrigIn (1 or 2) to use on the MFLI.
            edge: Trigger edge; one of "rising", "falling", "both".
            endless: If True, run continuous acquisition (advanced use).
            count: Number of grids to acquire in single-shot mode (endless=False).
        """
        # Normalize dependents list
        
        if isinstance(dependent, Sequence):
            self.dependents = [dep.zi_node for dep in dependent]
        else:
            self.dependents = [dependent.zi_node]

        self.daq.setInt(f"/{self.serial}/demods/0/enable", 1)
        self.daq.setDouble(f"/{self.serial}/demods/0/timeconstant", float(delay))

        self.daq_module.finish()
        self.daq_module.unsubscribe("*")
        self.daq_module.set("clearhistory", 1)

        self.daq_module.set(
            "triggernode", f"/{self.serial}/demods/0/sample.TrigIn{int(input_trigger)}"
        )
        edge_map = {"rising": 1, "falling": 2, "both": 3}
        self.daq_module.set("edge", edge_map.get(edge, 1))

        self.daq_module.set("endless", 1 if endless else 0)
        if not endless:
            self.daq_module.set("count", int(count))

        if isinstance(num, Sequence) and not isinstance(num, (str, bytes)):
            rows, cols = int(num[0]), int(num[1])
        else:
            rows, cols = 1, int(num)

        self.daq_module.set("grid/rows", rows)
        self.daq_module.set("grid/cols", cols)

        self.daq_module.set("duration", float(delay) * cols)

        self.daq_module.set("holdoff/time", max(0.0, float(delay) * (cols - 0.5)))
        self.daq_module.set("delay", 0.0)

        self._subs = []
        for dep in self.dependents:
            path = f"/{self.serial}/{dep}"
            self.daq_module.subscribe(path)
            self._subs.append(path)

        self.daq_module.execute()

    def fetch(self, *, timeout: float = 15.0) -> list[np.ndarray]:
        """
        Wait for DAQ completion and return a list of numpy arrays,
        one for each subscribed path in self._subs.
        """
        t0 = time()
        while not self.daq_module.finished():
            sleep(0.05)
            if time() - t0 > timeout:
                break

        result = self.daq_module.read()
        self.daq_module.finish()

        arrays: list[np.ndarray] = []
        for dep in self.dependents:
            dep_split = dep.split("/")
            data = result[self.serial][dep_split[0]][dep_split[1]][dep_split[2]][0][
                "value"
            ]
            arrays.append(np.array(data).flatten())

        return arrays


class NodeKeysightDMM(BufferedNodeBase):
    def __init__(
        self, dependent: Union[Parameter, Sequence[Parameter]] = None, *args, **kwargs
    ) -> None:
        super().__init__(inst=None, *args, **kwargs)

    def register_dependent(
        self, num_points: int, step_time: int | float, input_trigger: int = 1
    ) -> None:
        """
        Register the measurement with the Keysight DMM

        Args:
            num_points (int): Number of points in the sweep
            step_time (int | float): Duration of the current measurement in seconds. Depends on the preceeding sweep's step size.
            input_trigger (int): Input trigger for the Keysight DMM. Can only be 1 (external trigger) or any other value for continuoust triggering
        """
        self.core.aperture_time(step_time)
        self.core.timetrace_dt(num_points * step_time)
        self.core.timetrace_npts(num_points)
        if input_trigger == 1:
            self.core.trigger.source("EXT")
        else:
            self.core.trigger.source("IMM")
        self.core.trigger.count("INF")
        self.core.trigger.delay(0.0)
        self.core.sample.count(1)
        self.core.sample.pretrigger_count(0)
        self.core.init_measurement()

    def fetch(self) -> list:
        """
        Fetch the measurement from the Keysight DMM
        Returns:
            list: List of the measured values
        """
        result = self.core.fetch()
        print(result)


class NodeQDAC2(BufferedNodeBase):
    def __init__(
        self,
        inst: Instrument,
        *args,
        **kwargs,
    ) -> None:
        """
        QDAC2 as a node in the buffered sweep tree

        Args:
            sweep (Sweep or Sequence[Sweep]): The sweep object to be used in the buffered sweep tree. Can be a 1D or 2D sweep.
            dependent (Union[Parameter, Sequence[Parameter]]): Dependent parameter to be measured. Only read_current_A is supported.
        """
        super().__init__(inst=inst, *args, **kwargs)
        self.contacts = {}
        self._trigger_width = 1e-4

    def register_sweep(
        self,
        sweep: Union[Sweep, Sequence[Sweep]],
        input_trigger: int = None,
        output_trigger: int = None,
        trigger_type: str = "ramp",
    ):
        """
        Register the 1D/2D buffered sweep with triggers for the QDAC2

        Args:
            sweep (Union[Sweep, Sequence[Sweep]]): qcutils Sweep or list of Sweeps
            input_trigger (int): input trigger for the QDAC2 (Optional, defaults to None)
            output_trigger (int): output trigger for the QDAC2 (Optional, defaults to None)
            trigger_type (str): Type of trigger for the QDAC2, only to be used for 2D sweeps. One of either "ramp" (trigger at the start of each ramp) or "step" (at each step). (Optional, defaults to "ramp" for 2D sweeps, "step" for 1D sweeps)

        Returns:
            num_points (int): Number of points in the sweep
            step_time (int | float): Duration of the innermost sweep in seconds. num_points * step_time = total time of the whole sweep sequence
        """
        self._process_sweeps(sweep)

        if trigger_type not in ["ramp", "step"]:
            raise ValueError('trigger_type must be either "ramp" or "step"')

        if len(self.sweeps) == 2:
            inner_sweep = sweep[1]
            outer_sweep = sweep[0]

            inner_voltages = inner_sweep.values
            outer_voltages = outer_sweep.values

            self.input_trigger = (
                {f"trigin_{input_trigger}": input_trigger} if input_trigger else None
            )
            self.output_trigger = (
                {f"trigout_{output_trigger}": output_trigger}
                if output_trigger
                else None
            )
            self.input_trigger_key = (
                f"trigin_{input_trigger}" if input_trigger else None
            )
            self.output_trigger_key = (
                f"trigout_{output_trigger}" if output_trigger else None
            )

            self.contacts = {
                inner_sweep.parameter[0].name: inner_sweep.parameter[
                    0
                ].instrument._channum,
                outer_sweep.parameter[0].name: outer_sweep.parameter[
                    0
                ].instrument._channum,
            }
            self.arrangement = self.core.arrange(
                contacts=self.contacts,
                output_triggers=self.output_trigger,
            )

            for trig in self.core.external_triggers:
                trig.width_s(self._trigger_width)

            if trigger_type == "ramp":
                num_points = self.num_tup
                self._qdac_sweep = self.arrangement.virtual_sweep2d(
                    inner_contact=inner_sweep.parameter[0].name,
                    outer_contact=outer_sweep.parameter[0].name,
                    inner_voltages=inner_voltages,
                    outer_voltages=outer_voltages,
                    start_sweep_trigger=self.input_trigger_key,
                    inner_step_time_s=self.delay,
                    outer_step_trigger=self.output_trigger_key,
                )
            else:
                num_points = self.num
                self._qdac_sweep = self.arrangement.virtual_sweep2d(
                    inner_contact=inner_sweep.parameter[0].name,
                    outer_contact=outer_sweep.parameter[0].name,
                    inner_voltages=inner_voltages,
                    outer_voltages=outer_voltages,
                    start_sweep_trigger=self.input_trigger_key,
                    inner_step_time_s=self.delay,
                    inner_step_trigger=self.output_trigger_key,
                )

            return trigger_type, num_points, self.delay

        elif len(self.sweeps) == 1:
            num_points = self.num
            sweep = self.sweeps[0]

            self.contacts = {}
            if isinstance(sweep.parameter, Sequence):
                assert len(sweep.parameter) == 1, (
                    "Only one parameter supported for 1D sweeps"
                )
                for param in sweep.parameter:
                    self.contacts[param.name] = param.instrument._channum
            else:
                self.contacts[sweep.parameter.name] = (
                    sweep.parameter.instrument._channum
                )

            self.input_trigger = (
                {f"trigin_{input_trigger}": input_trigger} if input_trigger else None
            )
            self.output_trigger = (
                {f"trigout_{output_trigger}": output_trigger}
                if output_trigger
                else None
            )
            self.input_trigger_key = (
                f"trigin_{input_trigger}" if input_trigger else None
            )
            self.output_trigger_key = (
                f"trigout_{output_trigger}" if output_trigger else None
            )

            self.arrangement = self.core.arrange(
                contacts=self.contacts,
                output_triggers=self.output_trigger,
            )

            for trig in self.core.external_triggers:
                trig.width_s(self._trigger_width)

            if trigger_type == "ramp":
                raise NotImplementedError(
                    "trigger_type 'ramp' not implemented for 1D sweeps"
                )

            self._qdac_sweep = self.arrangement.virtual_sweep(
                contact=list(self.contacts.keys())[0],
                voltages=self.sweeps[0].values,
                start_sweep_trigger=self.input_trigger_key,
                step_time_s=self.delay,
                step_trigger=self.output_trigger_key,
            )

            return trigger_type, num_points, self.delay
        else:
            raise NotImplementedError("Only 1D and 2D sweeps are supported")

    def run_sweep(self):
        self._qdac_sweep.start()
        if self.toplevel:
            sleep((self.num + 1) * self.delay)

    def register_dependent(self, step_time: float):
        """
        Register the measurement with the QDAC2

        Args:
            step_time (int or float): Duration of the current measurement in seconds. Depends on the preceeding sweep's step size.
        """
        # if the device has sweeps set, then trigger by internal trigger
        # if the device is acting as an end node, then trigger by external trigger
        for dependent in self.dependents:
            assert dependent.name == "read_current_A", (
                "Only read_current_A is supported as a dependent for QDAC2"
            )
        if self.sweepnode:
            for dependent in self.dependents:
                dependent.underlying_instrument().clear_measurements()
                meas = dependent.underlying_instrument().measurement(
                    aperture_s=step_time
                )
                meas.start_on(
                    self.arrangement.get_trigger_by_name(list(self.triggers.keys())[0])
                )
        elif self.endnode:
            raise NotImplementedError("End node not implemented for QDAC2")
