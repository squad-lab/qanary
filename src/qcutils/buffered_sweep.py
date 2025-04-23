# %%
from qcodes.parameters import Parameter
from qcodes_contrib_drivers.drivers.QDevil.QDAC2 import QDac2
from drivers.squad.helpers.helpers import Lockin
from qcodes_contrib_drivers.drivers.QDevil.QDAC2 import QDac2
from drivers.squad.helpers.helpers import Lockin
from qcodes.instrument_drivers.Keysight import Keysight34461A
from typing import Sequence, Union
import zhinst.core

from qcutils.sweep import Sweep


class BufferedNodeBase:
    def __init__(
        self, sweep: Sweep, dependent: Union[Parameter, Sequence[Parameter]]
    ) -> None:
        """
        Base class for buffered sweep nodes. This class is a node of the buffered sweep tree. Can be used in three ways:
        1. As a node in a buffered sweep tree with a sweep and a dependent, where the device is both sweeping and measuring.
        2. As a node in a buffered sweep tree with a sweep and no dependent, where the device is only sweeping.
        3. As a node in a buffered sweep tree with no sweep and a dependent, where the device is only measuring.

        Args:
            sweep (Sweep): The sweep object to be used in the buffered sweep tree.
            dependent (BufferedDependent): The dependent object to be used in the buffered sweep tree.
        """
        self.buffered = True
        self.toplevel = False

        if sweep:
            self.sweep = sweep
            self.parameters = sweep.parameter
            instruments = [parameter.instrument for parameter in self.parameters]
            assert len(set(instruments)) == 1, (
                "All parameters of the buffered sweep must be from the same instrument"
            )
            assert len(set(instruments)) == 1, (
                "All parameters of the buffered sweep must be from the same instrument"
            )

            self.core = instruments[0]
            self.core.__init__()

            self.start = sweep.start
            self.stop = sweep.stop
            self.num = sweep.num
            self.step = sweep.step
            self.delay = sweep.delay
            self.start_delay = sweep.start_delay

        else:
            self.endnode = True

        if dependent:
            if not isinstance(dependent, Sequence):
                self.dependents = [dependent]
            else:
                self.dependents = dependent
            instruments = [dependent.instrument for dependent in self.dependents]
            assert len(set(instruments)) == 1, (
                "All dependents of the buffered node must be from the same instrument"
            )
            assert len(set(instruments)) == 1, (
                "All dependents of the buffered node must be from the same instrument"
            )

            self.core = instruments[0]
            self.core.__init__()

        else:
            self.sweepnode = True

        if not sweep and not dependent:
            raise ValueError(
                "BufferedNodeBase must have either a sweep or a dependent or both"
            )


class NodeMFLI(BufferedNodeBase):
    def __init__(
        self,
        dependent: Union[Parameter, Sequence[Parameter]],
        *args,
        **kwargs,
    ) -> None:
        """
        MFLI as a node in the buffered sweep tree
        This will be an end node in the buffered sweep tree. The MFLI will be used to measure the dependent parameter.

        Args:
            dependent (BufferedDependent): Dependent parameter to be measured
            trigger (int | float): Trigger input for the MFLI. Either 1 or 2.
        """
        super().__init__(sweep=None, dependent=dependent, *args, **kwargs)
        # self.session = self.core.session
        self.serial = self.core.serial
        self.daq = zhinst.core.ziDAQServer("127.0.0.1", 8004, 6)
        self.daq_module = self.daq.dataAcquisitionModule()
        self.daq_module.set("preview", 1)
        self.daq_module.set("device", self.serial)
        self.daq_module.set("type", 6)
        self.daq_module.set("endless", 1)
        self.daq_module.set("grid/mode", 2)

    def register_dependent(
        self, num_points: int, step_time: int | float, input_trigger: int = 1
        self, num_points: int, step_time: int | float, input_trigger: int = 1
    ) -> None:
        """
        Register the measurement with the MFLI

        Args:
            num_points (int): Number of points in the sweep
            num_points (int): Number of points in the sweep
            step_time (int | float): Duration of the current measurement in seconds. Depends on the preceeding sweep's step size.
            input_trigger (int): Input trigger for the MFLI. Can only be 1 or 2.
            input_trigger (int): Input trigger for the MFLI. Can only be 1 or 2.
        """
        self.daq_module.set(
            f"triggernode/{self.serial}/demods/0/sample.TrigIn{input_trigger}"
        )  # needs to be changed to allow for arbitrary trigins on LIA side
        self.daq_module.finish()
        self.daq_module.unsubscribe("*")

        self.daq.setDouble("/" + self.serial + "/demods/0/timeconstant", step_time)
        self.daq_module.set("grid/cols", num_points)
        self.daq_module.set("grid/cols", num_points)
        self.daq_module.set(
            "duration", dependent.parameter.instrument.timeconstant * num_points
            "duration", dependent.parameter.instrument.timeconstant * num_points
        )

        for dependent in self.dependents:
            self.daq_module.subscribe(
                f"/{self.serial}{dependent.parameter.zi_node}{dependent._values[0]}.avg"
            )

    def fetch(self):
        """
        Fetch the measurement from the Zurich Instruments MFLI
        Returns:
            list: List of the measured values from all dependents
        """
        result = self.daq_module.read()
        print(result)
        # result[self.serial]
        # return result


class NodeKeysightDMM(BufferedNodeBase):
    def __init__(
        self, dependent: Union[Parameter, Sequence[Parameter]], *args, **kwargs
    ) -> None:
        super().__init__(sweep=None, dependent=dependent, *args, **kwargs)

    def register_dependent(
        self, num_points: int, step_time: int | float, input_trigger: int = 1
        self, num_points: int, step_time: int | float, input_trigger: int = 1
    ) -> None:
        """
        Register the measurement with the Keysight DMM

        Args:
            num_points (int): Number of points in the sweep
            num_points (int): Number of points in the sweep
            step_time (int | float): Duration of the current measurement in seconds. Depends on the preceeding sweep's step size.
            input_trigger (int): Input trigger for the Keysight DMM. Can only be 1 (external trigger) or any other value for continuoust triggering
            input_trigger (int): Input trigger for the Keysight DMM. Can only be 1 (external trigger) or any other value for continuoust triggering
        """

        self.core.aperture_time(step_time)
        self.core.timetrace_dt(num_points * step_time)
        self.core.timetrace_npts(num_points)
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
        # return result


class NodeQDAC2(BufferedNodeBase):
    def __init__(
        self,
        sweep: Union[Sweep, Sequence[Sweep]],
        dependent: Union[Parameter, Sequence[Parameter]],
        *args,
        **kwargs,
    ) -> None:
        """
        QDAC2 as a node in the buffered sweep tree

        Args:
            sweep (Sweep or Sequence[Sweep]): The sweep object to be used in the buffered sweep tree. Can be a 1D or 2D sweep.
            dependent (Union[Parameter, Sequence[Parameter]]): Dependent parameter to be measured. Only read_current_A is supported.
        """
        super().__init__(sweep=sweep, dependent=dependent, *args, **kwargs)
        self.contacts = {}

    def register_sweep(
        self,
        sweep: Union[Sweep, Sequence[Sweep]],
        input_trigger: int = None,
        output_trigger: int = None,
    ):
        """
        Register the 1D/2D buffered sweep with triggers for the QDAC2

        Args:
            sweep (Union[Sweep, Sequence[Sweep]]): qcutils Sweep or list of Sweeps
            input_trigger (int): input trigger for the QDAC2 (Optional, defaults to None)
            output_trigger (int): output trigger for the QDAC2 (Optional, defaults to None)

        Returns:
            num_points (int): Number of points in the sweep
            step_time (int | float): Duration of the innermost sweep in seconds. num_points * step_time = total time of the whole sweep sequence
        """
        if isinstance(sweep, Sequence):
            # 2D sweep
            assert len(sweep) <= 2, "Maximum 2D sweep supported"
            for sw in sweep:
                assert len(sw.parameter) == 1, (
                    "Only one parameter per 2D sweep loop supported"
                )
                assert len(sw.parameter) == 1, (
                    "Only one parameter per 2D sweep loop supported"
                )
            inner_sweep = sweep[0]
            outer_sweep = sweep[1]
            inner_voltages = inner_sweep.values
            outer_voltages = outer_sweep.values
            inner_step_time_s = inner_sweep.delay
            outer_step_time_s = outer_sweep.delay
            num_points = inner_sweep.num * outer_sweep.num

            assert inner_step_time_s * inner_sweep.num == outer_step_time_s, (
                "Total time of the inner sweep must be equal to the outer sweep step time"
            )
            assert inner_step_time_s * inner_sweep.num == outer_step_time_s, (
                "Total time of the inner sweep must be equal to the outer sweep step time"
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

            self.dims = 2

            self.contacts = {
                inner_sweep.parameter[0].name: inner_sweep.parameter[0]
                inner_sweep.parameter[0].name: inner_sweep.parameter[0]
                .underlying_instrument()
                ._channum,
                outer_sweep.parameter[0].name: outer_sweep.parameter[0]
                outer_sweep.parameter[0].name: outer_sweep.parameter[0]
                .underlying_instrument()
                ._channum,
            }
            self.arrangement = self.core.arrange(
                contacts=self.contacts,
                output_triggers=self.output_trigger,
            )
            self._qdac_sweep = self.arrangement.virtual_sweep2d(
                inner_contact=inner_sweep.parameter[0].name,
                outer_contact=outer_sweep.parameter[0].name,
                inner_voltages=inner_voltages,
                outer_voltages=outer_voltages,
                inner_step_time_s=inner_step_time_s,
                outer_step_time_s=outer_step_time_s,
                inner_step_trigger=self.output_trigger_key,
                start_trigger=self.input_trigger_key,
            )

            return num_points, inner_step_time_s
        else:
            # 1D sweep
            # virtual detune for multiparameter sweep
            start = sweep.start
            stop = sweep.stop
            num_points = sweep.num
            step_time = sweep.delay
            self.dims = 1

            self.contacts = {}
            for param in sweep.parameter:
                self.contacts[param.name] = param.underlying_instrument()._channum

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

            self._qdac_sweep = self.arrangement.virtual_detune(
                contacts=list(self.contacts.keys()),
                start_V=[start] * len(self.contacts),
                stop_V=[stop] * len(self.contacts),
                steps=num_points,
                step_trigger=self.output_trigger_key,
                start_trigger=self.input_trigger_key,
                step_time_s=step_time,
                repititions=1,
            )
            return num_points, step_time

    def run_sweep(self):
        """
        Run the buffered sweep.
        """
        self._qdac_sweep.start()

    def register_dependent(self, step_time: int | float):
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
            assert dependent.name == "read_current_A", (
                "Only read_current_A is supported as a dependent for QDAC2"
            )

        if self.sweepnode:
            # internal trigger
            for dependent in self.dependents:
                dependent.underlying_instrument().clear_measurements()
                meas = dependent.underlying_instrument().measurement(
                    aperture_s=step_time
                )
                meas.start_on(
                    self.arrangement.get_trigger_by_name(list(self.triggers.keys())[0])
                )
        elif self.endnode:
            # external trigger
            raise NotImplementedError("End node not implemented for QDAC2")


# %%
dac = QDac2("dac", "localhost")
mfli = Lockin("mfli", "localhost")

sw = Sweep(dac.ch18.dc_constant_V, start=0, stop=1, num=101, delay=1e-2)
buffered_sweep = {
    "type": "buffered",
    "sw1": {
        "instrument": NodeQDAC2,
        "sweep": sw,
        "trig_out": 3,
        "nodes": {
            "sw2": {
                "instrument": NodeMFLI,
                "dependents": [mfli.auxins[0].value],
                "trig_in": 1,
            },
        },
    },
}


def _parse_bufsweep_tree(
    buffered_sweep,
    parent=None,
    toplevel=None,
    num_points: int = 0,
    step_time: float = 0.0,
    buffered_sweep,
    parent=None,
    toplevel=None,
    num_points: int = 0,
    step_time: float = 0.0,
):
    if parent is None:
    if parent is None:
        assert buffered_sweep["type"] == "buffered"
        buffered_sweep.pop("type")
        assert len(buffered_sweep) == 1, "Only one toplevel sweep allowed"
        toplevel = list(buffered_sweep.keys())[0]
        toplevel = list(buffered_sweep.keys())[0]

    for node_idx, node in enumerate(buffered_sweep):
        inst = node["instrument"](**node["extra_parameters"])
        if node["sweep"]:
            num_points_new, step_time = inst.register_sweep(
                node["sweep"],
                output_trigger=node["trig_out"],
                input_trigger=node["trig_in"],
            )
            num_points *= num_points_new
            # Start triggered sweeps if node has parent node.
            # Start toplevel node sweep later, which triggers the child node sweeps
            if parent:
                inst.run_sweep()
        if node["dependent"]:
            inst.register_dependent(
                node["dependent"],
                num_points=num_points,
                step_time=step_time,
                input_trigger=node["trig_in"],
            )
        inst = node["instrument"](**node["extra_parameters"])
        if node["sweep"]:
            num_points_new, step_time = inst.register_sweep(
                node["sweep"],
                output_trigger=node["trig_out"],
                input_trigger=node["trig_in"],
            )
            num_points *= num_points_new
            # Start triggered sweeps if node has parent node.
            # Start toplevel node sweep later, which triggers the child node sweeps
            if parent:
                inst.run_sweep()
        if node["dependent"]:
            inst.register_dependent(
                node["dependent"],
                num_points=num_points,
                step_time=step_time,
                input_trigger=node["trig_in"],
            )
        try:
            _parse_bufsweep_tree(
                buffered_sweep[node]["nodes"],
                parent=node,
                toplevel=toplevel,
                num_points=num_points,
                step_time=step_time,
                parent=node,
                toplevel=toplevel,
                num_points=num_points,
                step_time=step_time,
            )
        except KeyError:
            pass
        if node_idx == len(buffered_sweep.keys()) - 1:
            return toplevel
            return toplevel


# %%
toplevel = _parse_bufsweep_tree(buffered_sweep)
toplevel.run_sweep()
# %%
toplevel = _parse_bufsweep_tree(buffered_sweep)
toplevel.run_sweep()
# %%
