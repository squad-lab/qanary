from functools import wraps
from typing import Any, Dict, List, Optional, Tuple
from qcutils.logger import get_logger

logger = get_logger(__name__)


def parse_bufsweep_tree(visitor):
    """
    Walk a buffered sweep tree (new schema) and call `visitor` at each node.

    Schema (enforced)
    -----------------
    - `buffered_sweep` is a **single node payload dict** (the root).
    - Each node `payload` is a dict that may contain:
        - "name":         optional str
        - "instrument":   device object for this node
        - "sweeps":       optional sweep specification(s) for sweep nodes
        - "dependent":    optional dependent spec(s) for measurement nodes
        - "nodes":        optional list[child_payload_dict]
    - Children must be a **list**. Each child may omit "name" (auto-named node1, node2, ...).

    Visitor contract
    ----------------
    The walker calls:

        visitor(
            node: str,
            payload: dict,
            *,
            parent: Optional[str],
            path: Tuple[str, ...],
            instrument: Any,
            toplevel: Optional[Any],
            num_points: int,
            step_time: float,
            state: Dict[str, Any],
        ) -> Optional[Dict[str, Any]]

    If the visitor returns a dict, the following keys (if present) are propagated
    down this node's subtree only:
        - "num_points" : int
        - "step_time"  : float
        - "toplevel"   : Any
        - "state"      : dict  (branch-local)
    """

    def _assert_payload(d: Dict[str, Any]) -> None:
        if not isinstance(d, dict):
            raise TypeError("Root must be a payload dict.")
        if not any(k in d for k in ("instrument", "nodes", "sweeps", "dependent")):
            raise ValueError(
                "Root payload must contain at least one of "
                "'instrument', 'nodes', 'sweeps', or 'dependent'."
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
            if toplevel_in is None:
                curr_toplevel = instrument
                instrument.toplevel = True
            else:
                curr_toplevel = toplevel_in

            curr_num_points = num_pts_in
            curr_step_time = step_t_in
            curr_state = state_in
            curr_path = (*path_in, node_name)

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

            if isinstance(ret, dict):
                curr_num_points = ret.get("num_points", curr_num_points)
                curr_step_time = ret.get("step_time", curr_step_time)
                curr_toplevel = ret.get("toplevel", curr_toplevel)
                if "state" in ret:
                    curr_state = ret["state"]

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

        root_name = _name(root_payload, 1, default_root=True)
        return walk_one(
            root_name, root_payload, None, toplevel, num_points, step_time, state, path
        )

    return _walk_tree


@parse_bufsweep_tree
def arm_instruments(
    node,
    payload,
    *,
    parent,
    path,
    instrument,
    num_points,
    step_time,
    state,
    **kwargs,
):
    """
    Configure instruments and propagate sweep values.
    """

    if "sweeps" in payload:
        trigger_type, points_new, inner_step = instrument.register_sweep(
            sweep=payload["sweeps"],
            input_trigger=payload.get("input_trigger"),
            output_trigger=payload.get("output_trigger"),
            trigger_type=payload.get("trigger_type", "step"),
        )
        num_points *= points_new
        step_time = inner_step
        if parent:
            instrument.run_sweep()

    if "dependent" in payload:
        instrument.register_dependent(
            dependent=payload["dependent"],
            num=num_points,
            delay=step_time,
            input_trigger=payload.get("input_trigger"),
        )

    state.setdefault("visited_paths", []).append(path)

    return {"num_points": num_points, "step_time": step_time, "state": state}


@parse_bufsweep_tree
def fetch_dependents_tree(
    node, payload, *, parent, path, instrument, state, only_leaves=True, **kwargs
):
    """
    Visitor to build a dependent-tree summary.
    """

    children = payload.get("nodes", [])
    is_leaf = not children
    has_fetch = hasattr(instrument, "fetch") and callable(instrument.fetch)
    should_fetch = has_fetch and (is_leaf if only_leaves else True)

    # global collector
    state.setdefault("dependent_tree", {})

    # track sweeps
    if "sweeps" in payload:
        state.setdefault("sweeps", [])
        state.setdefault("sweep_shape", [])
        for sw in payload["sweeps"]:
            state["sweeps"].append(sw)
            state["sweep_shape"].append(sw.values.shape[0])

    # collect dependent metadata
    if should_fetch:
        sweep_shape = tuple(state.get("sweep_shape", []))
        sweeps = tuple(state.get("sweeps", []))

        for dep in payload.get("dependent", []):
            state["dependent_tree"][dep] = {
                "sweep_shape": sweep_shape,
                "sweeps": sweeps,
            }


@parse_bufsweep_tree
def fetch_results(
    node, payload, *, parent, path, instrument, state, only_leaves=True, **kwargs
):
    """
    Visitor to build a dependent-tree summary.
    """

    children = payload.get("nodes", [])
    is_leaf = not children
    has_fetch = hasattr(instrument, "fetch") and callable(instrument.fetch)
    should_fetch = has_fetch and (is_leaf if only_leaves else True)

    # global collector
    state.setdefault("results_tree", {})

    # track sweeps
    if "sweeps" in payload:
        state.setdefault("sweeps", [])
        state.setdefault("sweep_shape", [])
        for sw in payload["sweeps"]:
            state["sweeps"].append(sw)
            state["sweep_shape"].append(sw.values.shape[0])

    # collect dependent metadata
    if should_fetch:
        sweep_shape = tuple(state.get("sweep_shape", []))
        sweeps = tuple(state.get("sweeps", []))

        try:
            result_arrays = instrument.fetch()
            for idx, dep in enumerate(payload.get("dependent", [])):
                state["results_tree"][dep] = {
                    "sweeps": sweeps,
                    "result": result_arrays[idx].reshape(sweep_shape)
                    if sweep_shape
                    else result_arrays[idx],
                }

        except Exception as exc:
            logger.error(f"[{'/'.join(path)}] fetch() failed: {exc}")
