# %%
from qcodes.parameters import Parameter
from qcodes_contrib_drivers.drivers.QDevil.QDAC2 import QDAC2
from qcodes.instrument_drivers.Keysight import Keysight34461A
from typing import Sequence, Union
from zhinst.qcodes import MFLI
import zhinst.core
import numpy as np

from qcutils.sweep import Sweep

# import numpy as np

# buffered_sweep = {
#     "type": "buffered",
#     "sw1": {
#         "sweep": np.array,
#         "extra_parameters": np.array,
#         "nodes": {
#             "sw2": {
#                 "sweep": np.array,
#                 "extra_parameters": np.array,
#                 "nodes": {
#                     "sw3": {
#                         "sweep": np.array,
#                         "extra_parameters": np.array,
#                     },
#                     "sw4": {
#                         "sweep": np.array,
#                         "extra_parameters": np.array,
#                     },
#                 },
#             },
#             "sw5": {
#                 "sweep": np.array,
#                 "extra_parameters": np.array,
#             },
#         },
#     },
# }


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
            assert (
                len(set(instruments)) == 1
            ), "All parameters of the buffered sweep must be from the same instrument"

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
            assert (
                len(set(instruments)) == 1
            ), "All dependents of the buffered node must be from the same instrument"

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
        input_trigger: int | float,
        *args,
        **kwargs,
    ):
        """
        MFLI as a node in the buffered sweep tree
        This will be an end node in the buffered sweep tree. The MFLI will be used to measure the dependent parameter.

        Args:
            dependent (BufferedDependent): Dependent parameter to be measured
            trigger (int | float): Trigger input for the MFLI. Either 1 or 2.
        """
        super().__init__(sweep=None, dependent=dependent, *args, **kwargs)
        #self.session = self.core.session
        self.serial = self.core.serial
        self.daq = zhinst.core.ziDAQServer('127.0.0.1', 8004, 6) 
        self.daq_module = self.daq.dataAcquisitionModule()
        self.input_trigger = input_trigger
        self.daq_module.set('preview', 1)
        self.daq_module.set('device', self.serial)
        self.daq_module.set('type', 6)
        self.daq_module.set('triggernode', '/'+self.serial+'/demods/0/sample.TrigIn1') #needs to be changed to allow for arbitrary trigins on LIA side
        self.daq_module.set('endless', 1)
        self.daq_module.set('grid/mode', 2)

    def register_dependent(self,sweep_points: int, step_time: int | float):
        """
        Register the measurement with the MFLI

        Args:
            max_duration (float): Maximum duration of the measurement in seconds. Depends on the preceeding sweep's step size.
        """
        self.daq_module.finish()
        self.daq_module.unsubscribe('*')
        
        self.daq.setDouble('/'+self.serial+'/demods/0/timeconstant', step_time)
        self.daq_module.set('grid/cols', sweep_points)
        self.daq_module.set('duration', dependent.parameter.instrument.timeconstant*sweep_points)

        for dependent in self.dependents:
            self.daq_module.subscribe('/'+self.serial+dependent.parameter.zi_node+dependent._values[0]'.avg')

    def fetch(self):
        """
        Fetch the measurement from the Zurich Instruments MFLI
        Returns:
            list: List of the measured values from all dependents
        """
        result = self.daq_module.read()
        print(result)
        #result[self.serial]


class NodeKeysightDMM(BufferedNodeBase):
    def __init__(
        self, dependent: Union[Parameter, Sequence[Parameter]], *args, **kwargs
    ) -> None:
        super().__init__(sweep=None, dependent=dependent, *args, **kwargs)

    def register_dependent(self, step_time: int | float) -> None:
        """
        Register the measurement with the Keysight DMM
        Args:
            step_time (int | float): Duration of the current measurement in seconds. Depends on the preceeding sweep's step size.
        """
        self.core.aperture_time(step_time / 2)
        self.core.timetrace_dt(step_time)
        self.core.timetrace_npts(1)
        self.core.trigger.source("EXT")
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
        return self.core.fetch()


class NodeQDAC2(BufferedNodeBase):
    def __init__(
        self,
        sweep: Union[Sweep, Sequence[Sweep]],
        dependent: Union[Parameter, Sequence[Parameter]],
        output_trigger: dict,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            sweep (Sweep or Sequence[Sweep]): The sweep object to be used in the buffered sweep tree. Can be a 1D or 2D sweep.
            dependent (Union[Parameter, Sequence[Parameter]]): Dependent parameter to be measured. Only read_current_A is supported.
            output_trigger (dict): Dictionary of output triggers. The keys are the names of the output triggers and the values are physical trigger port numbers.
        """
        super().__init__(sweep=sweep, dependent=dependent, *args, **kwargs)
        self.contacts = {}
        if isinstance(sweep, Sequence):
            # 2D sweep
            assert len(sweep) <= 2, "Maximum 2D sweep supported"
            for sw in sweep:
                assert (
                    len(sw.parameter) == 1
                ), "Only one parameter per 2D sweep loop supported"
            inner_sweep = sweep[0]
            outer_sweep = sweep[1]
            inner_voltages = inner_sweep.values
            outer_voltages = outer_sweep.values
            inner_step_time_s = inner_sweep.delay
            outer_step_time_s = outer_sweep.delay
            self.num_points = inner_sweep.num * outer_sweep.num
            self.triggers = output_trigger
            self.dims = 2

            self.contacts = {
                inner_sweep.parameter[0]
                .name: inner_sweep.parameter[0]
                .underlying_instrument()
                ._channum,
                outer_sweep.parameter[0]
                .name: outer_sweep.parameter[0]
                .underlying_instrument()
                ._channum,
            }
            self.arrangement = self.core.arrange(
                contacts=self.contacts,
                output_triggers=self.triggers,
            )
            self._qdac_sweep = self.arrangement.virtual_sweep2d(
                inner_contact=inner_sweep.parameter[0].name,
                outer_contact=outer_sweep.parameter[0].name,
                inner_voltages=inner_voltages,
                outer_voltages=outer_voltages,
                inner_step_time_s=inner_step_time_s,
                outer_step_time_s=outer_step_time_s,
                inner_step_trigger=list(self.triggers.keys())[0],
            )

        else:
            # 1D sweep
            # virtual detune for multiparameter sweep
            start = sweep.start
            stop = sweep.stop
            self.num_points = sweep.num
            delay = sweep.delay
            self.dims = 1

            self.contacts = {}
            for param in sweep.parameter:
                self.contacts[param.name] = param.underlying_instrument()._channum

            self.triggers = output_trigger
            self.arrangement = self.core.arrange(
                contacts=self.contacts,
                output_triggers=self.triggers,
            )

            self._qdac_sweep = self.arrangement.virtual_detune(
                contacts=list(self.contacts.keys()),
                start_V=[start] * len(self.contacts),
                stop_V=[stop] * len(self.contacts),
                steps=self.num_points,
                step_trigger=list(self.triggers.keys())[0],
                step_time_s=delay,
                repititions=1,
            )

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
            assert (
                dependent.name == "read_current_A"
            ), "Only read_current_A is supported as a dependent for QDAC2"

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


def _parse_bufsweep_tree(
    buffered_sweep, parent_node=None, toplevel_node=None, indent=" "
):
    indent += " "
    if parent_node is None:
        assert buffered_sweep["type"] == "buffered"
        buffered_sweep.pop("type")
        assert len(buffered_sweep) == 1, "Only one toplevel sweep allowed"
        toplevel_node = buffered_sweep[list(buffered_sweep.keys())[0]]

    for node_idx, node in enumerate(buffered_sweep):
        
        # # register node, sweep, and extra_parameters
        print(
            f"{indent}Node: {node.sweep.parameter}: {node.sweep.parameter.instrument.name}"
        )
        # print(f"{indent}{node}")
        node.sweep = buffered_sweep[node]["sweep"]
        node.extra_parameters = buffered_sweep[node]["extra_parameters"]
        node.subscribe()
        try:
            _parse_bufsweep_tree(
                buffered_sweep[node]["nodes"],
                parent_node=node,
                toplevel_node=toplevel_node,
                indent=indent,
            )
        except KeyError:
            pass
        if node_idx == len(buffered_sweep.keys()) - 1:
            return toplevel_node["sweep"], toplevel_node["extra_parameters"]


# %%
