# %%
from functools import wraps
from time import sleep, time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
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

    def _process_dependents(
        self, dependent: Parameter | Sequence[Parameter] | str | Sequence[str]
    ):
        """
        Process dependents

        Args:
            dependent (Parameter | Sequence[Parameter]): The dependent object to be used in the buffered sweep tree.
        """

        if not isinstance(dependent, Sequence):
            self.dependents = [dependent]
        else:
            self.dependents = dependent
        #     instruments = [dependent.instrument for dependent in self.dependents]
        # assert len(set(instruments)) == 1, (
        #     "All dependents of the buffered node must be from the same instrument"
        # )

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

        # Underlying LabOne objects
        self.core = self.core.core
        self.serial = self.core.serial
        self.daq = self.core.session.daq_server

        # DAQ module (single instance per node)
        self.daq_module = self.daq.dataAcquisitionModule()
        self.daq_module.set("device", self.serial)
        self.daq_module.set("type", 6)  # hardware trigger
        self.daq_module.set("grid/mode", 2)  # exact grid

        # Keep track of what we subscribed to (useful when parsing results)
        self._subs: list[str] = []

    def register_dependent(
        self,
        dependent: str | Sequence[str],
        num: int | Sequence[int],
        delay: float,
        input_trigger: int = 1,
        *,
        edge: str = "rising",  # "rising" | "falling" | "both"
        endless: bool = False,  # prefer single-shot
        count: int = 1,  # number of grids to acquire (single-shot)
    ) -> None:
        """
        Configure DAQ for a hardware-triggered, exact-grid acquisition and **start** it.

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
        if isinstance(dependent, Sequence) and not isinstance(dependent, (str, bytes)):
            self.dependents = list(dependent)
        else:
            self.dependents = [str(dependent)]

        # Ensure the demodulator is enabled for streaming
        self.daq.setInt(f"/{self.serial}/demods/0/enable", 1)
        self.daq.setDouble(f"/{self.serial}/demods/0/timeconstant", float(delay))

        # Clean previous runs and history
        self.daq_module.finish()
        self.daq_module.unsubscribe("*")
        self.daq_module.set("clearhistory", 1)

        # Trigger node & edge (required for type=6)
        self.daq_module.set(
            "triggernode", f"/{self.serial}/demods/0/sample.TrigIn{int(input_trigger)}"
        )
        edge_map = {"rising": 1, "falling": 2, "both": 3}
        self.daq_module.set("edge", edge_map.get(edge, 1))

        # Acquisition mode
        self.daq_module.set("endless", 1 if endless else 0)
        if not endless:
            self.daq_module.set("count", int(count))

        # Grid shape
        if isinstance(num, Sequence) and not isinstance(num, (str, bytes)):
            assert len(num) == 2, "For 2D, pass (rows, cols)"
            rows, cols = int(num[0]), int(num[1])
        else:
            rows, cols = 1, int(num)
        self.daq_module.set("grid/rows", rows)
        self.daq_module.set("grid/cols", cols)

        # Duration per row (exact mode): ~ step_time * cols
        self.daq_module.set("duration", float(delay) * cols)

        # Optional: holdoff to avoid re-triggering too soon (row-level)
        self.daq_module.set("holdoff/time", max(0.0, float(delay) * (cols - 0.5)))
        self.daq_module.set("delay", 0.0)  # relative delay to the trigger edge

        # Subscribe to each dependent (explicit signal paths like ".../sample.r")
        self._subs = []
        for dep in self.dependents:
            path = f"/{self.serial}/{dep}"
            self.daq_module.subscribe(path)
            self._subs.append(path)

        # **ARM** the DAQ: without execute(), read() will be empty.
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
        print("something")
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
        self._trigger_width = 1e-4
        # for trig in self.core.external_triggers:
        #     trig.width_s(10e-3)

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

            for trig in self.core.external_triggers:
                trig.width_s(self._trigger_width)
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
            num_points = self.num

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

            for trig in self.core.external_triggers:
                trig.width_s(self._trigger_width)
            if trigger_type == "ramp":
                raise NotImplementedError(
                    "trigger_type 'ramp' not implemented for 1D sweeps"
                )
            else:
                self._qdac_sweep = self.arrangement.virtual_sweep(
                    contact=list(self.contacts.keys())[0],
                    voltages=self.sweeps[0].values,
                    start_sweep_trigger=self.input_trigger_key,
                    step_time_s=self.delay,
                    step_trigger=self.output_trigger_key,
                )
            return trigger_type, num_points, self.delay

    def run_sweep(self):
        """
        Run the buffered sweep.
        """
        self._qdac_sweep.start()
        if self.toplevel:
            print((self.num + 1) * self.delay)
            sleep((self.num + 1) * self.delay)

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
Instrument.close_all()
mfli1 = Lockin(name="mfli1", address="192.168.0.104", serial="DEV7128")
dac = QDac2("dac", "TCPIP0::qdevil_dac_1.lab.squad-lab.org::5025::SOCKET")

sw = Sweep(dac.ch24.dc_constant_V, start=0, stop=0.1, num=101, delay=0.01)
buffered_sweep = {
    "instrument": NodeQDAC2(inst=dac),
    "sweep": sw,
    "output_trigger": 1,
    "trigger_type": "step",  # single, ramp or every
    "nodes": [
        {
            "instrument": NodeMFLI(inst=mfli1),
            "dependent": ["demods/0/sample.r"],
            "input_trigger": 1,
        }
    ],
}


def parse_bufsweep_tree(visitor):
    """
    Walk a buffered sweep tree (new schema, no 'type' header) and call `visitor` at each node.

    Schema (enforced)
    -----------------
    - `buffered_sweep` is a **single node payload dict** (the root).
    - Each node `payload` is a dict that may contain:
        - "name":         optional str
        - "instrument":   device object for this node
        - "sweep":        optional sweep spec for sweep nodes
        - "dependent":    optional dependent spec for measurement nodes
        - "nodes":        optional list[child_payload_dict]
    - Children must be a **list**. Each child may omit "name" (auto-named node1, node2, ...).

    Visitor contract
    ----------------
    The walker calls:

        visitor(
            node: str,                # current node name (auto if missing)
            payload: dict,            # the node payload dict
            *,
            parent: Optional[str],    # parent node name, or None at root
            path: Tuple[str, ...],    # ("root" or provided name, ..., child_name)
            instrument: Any,          # payload.get("instrument")
            toplevel: Optional[Any],  # instrument at root (first seen)
            num_points: int,          # branch-local sweep points so far
            step_time: float,         # branch-local step time so far
            state: Dict[str, Any],    # branch-local mutable state (copy per sibling)
        ) -> Optional[Dict[str, Any]]

    If the visitor returns a dict, the following keys (if present) are **propagated
    down this node's subtree only**:
        - "num_points": int
        - "step_time": float
        - "toplevel": Any
        - "state": dict   (replaces the branch-local state for children)

    The walker returns the discovered `toplevel` instrument after traversal.
    """

    def _assert_payload(d: Dict[str, Any]) -> None:
        if not isinstance(d, dict):
            raise TypeError("Root must be a payload dict.")
        # Heuristic: must look like a node payload
        if not any(k in d for k in ("instrument", "nodes", "sweep", "dependent")):
            raise ValueError(
                "Root payload must contain at least one of "
                "'instrument', 'nodes', 'sweep', or 'dependent'."
            )

    def _children_list(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        ch = payload.get("nodes", [])
        if ch is None:
            return []
        if not isinstance(ch, list):
            raise TypeError("'nodes' must be a list of child payload dicts.")
        return ch

    def _name(payload: Dict[str, Any], idx: int, default_root: bool) -> str:
        n = payload.get("name")
        if isinstance(n, str) and n.strip():
            return n
        return "root" if default_root else f"node{idx}"

    @wraps(visitor)
    def _walk_tree(
        root_payload: Dict[str, Any],
        *,
        parent: Optional[str] = None,
        toplevel: Optional[Any] = None,
        num_points: int = 1,
        step_time: float = 0.0,
        state: Optional[Dict[str, Any]] = None,
        path: Tuple[str, ...] = (),
    ):
        _assert_payload(root_payload)
        if state is None:
            state = {}

        # Depth-first traversal over this payload and its children
        def walk_one(
            node_name: str,
            payload: Dict[str, Any],
            parent_name: Optional[str],
            toplevel_in: Optional[Any],
            num_pts_in: int,
            step_t_in: float,
            state_in: Dict[str, Any],
            path_in: Tuple[str, ...],
        ) -> Optional[Any]:
            instrument = payload.get("instrument")
            curr_toplevel = toplevel_in if toplevel_in is not None else instrument

            # per-node copies (siblings isolation)
            curr_num_points = num_pts_in
            curr_step_time = step_t_in
            curr_state = dict(state_in)
            curr_path = (*path_in, node_name)

            # visit current node
            ret = visitor(
                node_name,
                payload,
                parent=parent_name,
                path=curr_path,
                instrument=instrument,
                toplevel=curr_toplevel,
                num_points=curr_num_points,
                step_time=curr_step_time,
                state=curr_state,
            )

            # branch-local propagation
            if isinstance(ret, dict):
                if "num_points" in ret:
                    curr_num_points = ret["num_points"]
                if "step_time" in ret:
                    curr_step_time = ret["step_time"]
                if "toplevel" in ret:
                    curr_toplevel = ret["toplevel"]
                if "state" in ret and isinstance(ret["state"], dict):
                    curr_state = ret["state"]

            # recurse into children list
            children = _children_list(payload)
            for i, child in enumerate(children, start=1):
                if not isinstance(child, dict):
                    raise TypeError("Each child must be a payload dict.")
                child_name = _name(child, i, default_root=False)
                walk_one(
                    child_name,
                    child,
                    node_name,
                    curr_toplevel,
                    curr_num_points,
                    curr_step_time,
                    curr_state,
                    curr_path,
                )

            return curr_toplevel

        # Kick off at root
        root_name = _name(root_payload, 1, default_root=True)
        return walk_one(
            root_name, root_payload, None, toplevel, num_points, step_time, state, path
        )

    return _walk_tree


@parse_bufsweep_tree
def arm_instruments(
    node, payload, *, parent, path, instrument, num_points, step_time, state, **kwargs
):
    """
    Arm and configure instruments in a buffered sweep tree.

    This visitor is designed to be used with `@parse_bufsweep_tree`.
    It configures instruments for each sweep node, propagates updated
    sweep parameters (`num_points`, `step_time`) to children, and
    accumulates traversal state.

    The traversal follows the structure of the buffered sweep tree:

        Root
        └── nodeA (instrument=awg0)
            ├── child1 (instrument=awg1)
            │   └── grandchild (instrument=awg2)
            └── child2 (instrument=daq0)

    Each node may define:
      - "sweep": sweep configuration for its instrument
      - "dependent": dependent parameter(s) to be registered
      - "nodes": nested children

    Args:
        payload (dict): The dictionary payload for the current node.
        parent (str | None): Parent node name, or None at the root.
        path (tuple[str, ...]): Full path of node keys from root to current.
        instrument (Any): The instrument object at this node.
        num_points (int): Number of sweep points accumulated so far.
        step_time (float): Sweep step time accumulated so far.
        state (dict): Mutable state bag passed down the branch.

    Returns:
        dict: A dictionary of updates for this branch. May include:
            - "num_points": updated number of points
            - "step_time": updated step time
            - "state": updated branch-local state

    Behavior:
        - If a "sweep" key is present:
            * Registers the sweep with the instrument
            * Updates num_points and step_time
            * If not at root, triggers the instrument sweep immediately
        - If a "dependent" key is present:
            * Registers dependent measurements with the instrument
        - Appends the current path to `state["visited_paths"]`
    """

    # Example: multiply points if this node defines a sweep
    if "sweep" in payload:
        _, points_new, step_time = instrument.register_sweep(
            sweep=payload["sweep"],
            output_trigger=payload["output_trigger"],
            input_trigger=payload["input_trigger"] if parent else None,
            trigger_type=payload.get("trigger_type", "ramp"),
        )
        num_points *= points_new
        if parent:
            instrument.run_sweep()

    # Example: register dependents
    if "dependent" in payload:
        instrument.register_dependent(
            dependent=payload["dependent"],
            num=num_points,
            delay=step_time,
            input_trigger=payload.get("input_trigger"),
        )

    # Example: collect full node paths for debugging / reporting
    state.setdefault("visited_paths", []).append(path)

    # Propagate updated values to this node's children only
    return {"num_points": num_points, "step_time": step_time, "state": state}


@parse_bufsweep_tree
def fetch_results(node, payload, *, path, instrument, **kwargs):
    """
    Fetch and return measurement results from already-instantiated instruments.

    Behavior:
      • If the node has an instrument with a callable `fetch()`, this visitor calls it.
      • By default it fetches at leaves only; to fetch at any node, pass
        `only_leaves=False` into the walker (see walker tweak below).
      • Returns a dict with {"path": tuple(path), "result": result} when it fetches.
      • Does NOT modify/propagate any state.

    Returns:
      dict | None:
          - {"path": tuple[str, ...], "result": Any} when a fetch occurs at this node
          - None otherwise
    """
    # Determine children & leaf-ness
    children = payload.get("nodes") if isinstance(payload, dict) else None
    is_leaf = not (isinstance(children, dict) and children)

    # Decide whether to fetch at this node
    only_leaves = kwargs.get("only_leaves", True)  # accepted via walker kwargs
    has_fetch = hasattr(instrument, "fetch") and callable(getattr(instrument, "fetch"))
    should_fetch = has_fetch and (is_leaf if only_leaves else True)

    if should_fetch:
        pretty_path = "/".join(path)
        try:
            result = instrument.fetch()
            print(f"[{pretty_path}] {result}")
            return {"path": tuple(path), "result": result}
        except Exception as exc:
            print(f"[{pretty_path}] fetch() failed: {exc}")
            return {"path": tuple(path), "result": exc}

    return None


# %%
toplevel = arm_instruments(buffered_sweep)
# %%
toplevel.run_sweep()

# %%
fetch_results(buffered_sweep)

# %%
