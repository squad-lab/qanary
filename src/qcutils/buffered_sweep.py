# %%
from typing import Sequence, Union

from qcodes.instrument import Instrument
from qcodes.parameters import Parameter
from qcodes_contrib_drivers.drivers.QDevil.QDAC2 import QDac2

from drivers.squad.helpers.helpers import Lockin
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

        # self.core.__init__()
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

    def _process_dependents(self, dependent: Parameter | Sequence[Parameter]):
        """
        Process dependents

        Args:
            dependent (Parameter | Sequence[Parameter]): The dependent object to be used in the buffered sweep tree.
        """
        if not isinstance(dependent, Sequence):
            self.dependents = [dependent]
        else:
            self.dependents = dependent
        instruments = [dependent.instrument for dependent in self.dependents]
        assert len(set(instruments)) == 1, (
            "All dependents of the buffered node must be from the same instrument"
        )

        self.sweepnode = False


class NodeMFLI(BufferedNodeBase):
    def __init__(
        self,
        inst: Instrument,
        *args,
        **kwargs,
    ) -> None:
        """
        MFLI as a node in the buffered sweep tree
        This will be an end node in the buffered sweep tree. The MFLI will be used to measure the dependent parameter.

        Args:
            inst (Instrument): MFLI instrument responsible for this node
        """
        super().__init__(inst=inst, *args, **kwargs)
        self.core = self.core.core
        self.serial = self.core.serial
        self.daq = self.core.session.daq_server
        self.daq_module = self.daq.dataAcquisitionModule()
        self.daq_module.set("preview", 1)
        self.daq_module.set("device", self.serial)
        self.daq_module.set("type", 6)
        self.daq_module.set("endless", 1)
        self.daq_module.set("grid/mode", 2)

    def register_dependent(
        self,
        dependent: Parameter | Sequence[Parameter],
        num: int | Sequence[int],
        delay: int | float,
        input_trigger: int = 1,
    ) -> None:
        """
        Register the measurement with the MFLI

        Args:
            num (int | float): Number of points in the sweep. Can be a single int or a tuple of two ints for 2D sweeps.
            delay (int | float): Duration of the current measurement in seconds. Depends on the preceeding sweep's step size.
            input_trigger (int): Input trigger for the MFLI. Can only be 1 or 2.
        """
        self._process_dependents(dependent)
        print(f"Registering dependents for {self.core.name}")
        self.daq_module.set(
            "triggernode", f"/{self.serial}/demods/0/sample.TrigIn{input_trigger}"
        )
        self.daq_module.finish()
        self.daq_module.unsubscribe("*")

        self.daq.setDouble("/" + self.serial + "/demods/0/timeconstant", delay)

        if isinstance(num, Sequence):
            assert len(num) == 2, "num can be a maximum of two ints for 2D sweeps"
            self.daq_module.set("grid/cols", num[1])
            self.daq_module.set("grid/rows", num[0])
            self.daq_module.set(
                "duration",
                delay * num[0] * num[1],
            )
            # holdoff < time for one row. Doesnt matter anymore if trigger is once per sweep or every step
            self.daq_module.set("holdoff/time", delay * (num[1] - 0.5))

        else:
            self.daq_module.set("grid/cols", num)
            self.daq_module.set(
                "duration",
                delay * num,
            )
            # holdoff < time for one row. Doesnt matter anymore if trigger is once per sweep or every step
            self.daq_module.set("holdoff/time", delay * (num - 0.5))

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
        self, dependent: Union[Parameter, Sequence[Parameter]] = None, *args, **kwargs
    ) -> None:
        super().__init__(sweep=None, dependent=dependent, *args, **kwargs)

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
        # return result


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
        super().__init__(inst=inst, *args, **kwargs)  # name = self.inst.name, adress =
        self.contacts = {}
        for trig in self.core.external_triggers:
            trig.width_s(10e-3)

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
            for sw in sweep:
                assert len(sw.parameter) == 1, (
                    "Only one parameter per 2D sweep loop supported"
                )

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
                inner_sweep.parameter[0].name: inner_sweep.parameter[0]
                .underlying_instrument()
                ._channum,
                outer_sweep.parameter[0].name: outer_sweep.parameter[0]
                .underlying_instrument()
                ._channum,
            }
            self.arrangement = self.core.arrange(
                contacts=self.contacts,
                output_triggers=self.output_trigger,
            )

            # send output trigger at each ramp start
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
            # send output trigger at each step
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
        else:
            # 1D sweep
            # virtual detune for multiparameter sweep
            start = sweep.values[0]
            stop = sweep.values[-1]
            num_points = self.num

            self.contacts = {}
            if isinstance(sweep.parameter, Sequence):
                for param in sweep.parameter:
                    self.contacts[param.name] = param.instrument._channum
            else:
                self.contacts[sweep.parameter.name] = (
                    sweep.parameter.instrument._channum
                )  # no clue what the _channum is supposed to do here
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

            if trigger_type == "ramp":
                raise NotImplementedError(
                    "trigger_type 'ramp' not implemented for 1D sweeps"
                )
            else:
                self._qdac_sweep = self.arrangement.virtual_detune(
                    contacts=list(self.contacts.keys()),
                    start_V=[start] * len(self.contacts),
                    end_V=[stop] * len(self.contacts),
                    steps=num_points,
                    step_trigger=self.output_trigger_key,
                    start_trigger=self.input_trigger_key,
                    step_time_s=self.delay,
                )
            return trigger_type, num_points, self.delay

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
# dac = QDac2("dac", "localhost")
# mfli = Lockin("mfli", "localhost")
Instrument.close_all()
mfli1 = Lockin(name="mfli1", address="192.168.0.104", serial="DEV7128")
dac = QDac2("dac", "TCPIP0::qdevil_dac_1.lab.squad-lab.org::5025::SOCKET")  #

sw = Sweep(dac.ch24.dc_constant_V, start=0, stop=0.01, num=101, delay=1e-2)
buffered_sweep = {
    "type": "buffered",
    "sw1": {
        "instrument": NodeQDAC2(inst=dac),
        "sweep": sw,
        "output_trigger": 1,
        "trigger_type": "step",  # single, ramp or every
        "nodes": {
            "sw2": {
                "instrument": NodeMFLI(inst=mfli1),
                "dependent": [mfli1.core.demods[0].sample["R"]],
                "input_trigger": 1,
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
):
    if not parent:
        assert buffered_sweep["type"] == "buffered"
        buffered_sweep.pop("type")
        assert len(buffered_sweep) == 1, "Only one toplevel sweep allowed"

    for node_idx, node in enumerate(buffered_sweep):
        inst = buffered_sweep[node]["instrument"]

        if not parent:
            toplevel = inst

        if "sweep" in buffered_sweep[node]:
            _, num_points_new, step_time = inst.register_sweep(
                sweep=buffered_sweep[node]["sweep"],
                output_trigger=buffered_sweep[node]["output_trigger"],
                input_trigger=buffered_sweep[node]["input_trigger"] if parent else None,
                trigger_type=buffered_sweep[node]["trigger_type"]
                if "trigger_type" in buffered_sweep[node]
                else "ramp",
            )

            num_points *= num_points_new
            # Start triggered sweeps if node has parent node.
            # Start toplevel node sweep later, which triggers the child node sweeps
            if parent:
                inst.run_sweep()

        if "dependent" in buffered_sweep[node]:
            inst.register_dependent(
                dependent=buffered_sweep[node]["dependent"],
                num=num_points,
                delay=step_time,
                input_trigger=buffered_sweep[node]["trig_in"],
            )
        try:
            _parse_bufsweep_tree(
                buffered_sweep[node]["nodes"],
                parent=node,
                toplevel=toplevel,
                num_points=num_points,
                step_time=step_time,
            )
        except Exception:
            pass
        if node_idx == len(buffered_sweep.keys()) - 1:
            return toplevel


# %%
toplevel = _parse_bufsweep_tree(buffered_sweep)
# %%
toplevel.run_sweep()
