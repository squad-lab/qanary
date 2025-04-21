# %%
from qcodes.parameters import Parameter
from qcodes_contrib_drivers.drivers.QDevil.QDAC2 import QDAC2
from qcodes.instrument_drivers.Keysight import Keysight34461A
from typing import Sequence, Union
from zhinst.qcodes import MFLI
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


def register_buffered_dependent():
    pass


class BufferedDependent:
    """
    Class to define a timed buffered measurement.
    """

    def __init__(self, dependent: Union[Parameter, Sequence[Parameter]]):
        self.buffered = True
        if isinstance(dependent, Sequence):
            instruments = [dep.instrument for dep in dependent]
            assert (
                len(set(instruments)) == 1
            ), "All elements of dependent must be from the same instrument"
            self.dependent = dependent
        else:
            self.dependent = [dependent]


class BufferedNodeBase:
    def __init__(self, sweep: Sweep, dependent: BufferedDependent) -> None:
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
        self, dependent: BufferedDependent, input_trigger: int | float, *args, **kwargs
    ):
        """
        MFLI as a node in the buffered sweep tree
        This will be an end node in the buffered sweep tree. The MFLI will be used to measure the dependent parameter.

        Args:
            dependent (BufferedDependent): Dependent parameter to be measured
            trigger (int | float): Trigger input for the MFLI. Either 1 or 2.
        """
        super().__init__(sweep=None, dependent=dependent, *args, **kwargs)
        self.session = self.core.session
        self.input_trigger = input_trigger

    def register_dependent(self, max_duration: float = 1e-3):
        """
        Register the measurement with the MFLI

        Args:
            max_duration (float): Maximum duration of the measurement in seconds. Depends on the preceeding sweep's step size.
        """
        num_cols = 100
        num_bursts = 1
        daq = self.session.modules.daq
        daq.device(self.core)
        daq.type(1)
        daq.edge(0)
        daq.grid.mode(2)
        daq.count(num_bursts)
        daq.duration(max_duration)
        daq.grid.cols(num_cols)
        for dependent in self.dependents:
            daq.subscribe(dependent)


class NodeKeysightDMM(BufferedNodeBase):
    def __init__(self, dependent: BufferedDependent, *args, **kwargs):
        super().__init__(sweep=None, dependent=dependent, *args, **kwargs)

    def register_dependent(self, max_duration: float = 1e-3):
        pass


class NodeQDAC2(BufferedNodeBase):
    def __init__(self, sweep: Union[Sweep, Sequence[Sweep]], dependent: BufferedDependent, output_triggers: dict, *args, **kwargs):
        super().__init__(sweep=sweep, dependent=dependent, *args, **kwargs)
        self.contacts = {}
        if isinstance(sweep, Sequence):
            # 2D sweep
            assert len(sweep) <= 2, "Maximum 2D sweep supported"
            for sw in sweep:
                assert len(sw.parameter) == 1, "Only one parameter per 2D sweep loop supported"
            inner_sweep = sweep[0]
            outer_sweep = sweep[1]
            inner_voltages = inner_sweep.values
            outer_voltages = outer_sweep.values
            inner_step_time_s = inner_sweep.delay
            outer_step_time_s = outer_sweep.delay

            self.output_triggers = output_triggers
            
            self.contacts = {
                inner_sweep.parameter[0].name: inner_sweep.parameter[0].underlying_instrument()._channum,
                outer_sweep.parameter[0].name: outer_sweep.parameter[0].underlying_instrument()._channum,
            }
            self.arrangement = self.core.arrange(
                contacts=self.contacts,
                output_triggers=self.output_triggers,
            )
            self._qdac_sweep = self.arrangement.virtual_sweep2d(
                inner_contact=inner_sweep.parameter[0].name,
                outer_contact=outer_sweep.parameter[0].name,
                inner_voltages=inner_voltages,
                outer_voltages=outer_voltages,
                inner_step_time_s=inner_step_time_s,
                outer_step_time_s=outer_step_time_s,
                inner_step_trigger=list(self.output_triggers.keys())[0],
                outer_step_trigger=list(self.output_triggers.keys())[1],
            )

        else:
            # 1D sweep
            # virtual detune for multiparameter sweep
            start = sweep.start
            stop = sweep.stop
            num = sweep.num
            delay = sweep.delay

            self.contacts = {}
            for param in sweep.parameter:
                self.contacts[param.name] = param.underlying_instrument()._channum
            
            self.output_triggers = output_triggers
            self.arrangement = self.core.arrange(
                contacts=self.contacts,
                output_triggers=self.output_triggers,
            )

            self._qdac_sweep = self.arrangement.virtual_detune(
                contacts=list(self.contacts.keys()),
                start_V=[start]*len(self.contacts),
                stop_V=[stop]*len(self.contacts),
                steps=num,
                step_trigger=list(self.output_triggers.keys()),
                step_time_s=delay,
                repititions=1
                )


    def run_sweep(self):
        """
        Run the buffered sweep.
        """
        self._qdac_sweep.start()

    def register_dependent(self, max_duration: float = 1e-3):
        pass


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