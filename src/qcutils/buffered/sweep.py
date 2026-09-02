from functools import wraps
from inspect import signature
from typing import Any, Dict, List, Optional, Tuple

from qcutils.logger import get_logger

logger = get_logger(__name__)


def _buffered_sweep_progress_info(root_payload: Dict[str, Any]) -> Tuple[int, float]:
    """
    Return the point count and estimated duration of a buffered block.

    Buffered sibling branches are driven concurrently, so one block is
    represented by its longest coordinate path instead of by summing every
    dependent. Instrument setup and result-fetch overhead are not included in
    the duration estimate.

    Args:
        root_payload (Dict[str, Any]): Root dictionary of the buffered sweep
            tree.

    Returns:
        Tuple[int, float]: The largest buffered point count and the estimated
            duration in seconds of the slowest branch.

    Raises:
        TypeError: If the root, a child, ``sweeps``, or ``nodes`` has an invalid
            type.

    """

    if not isinstance(root_payload, dict):
        raise TypeError("Buffered sweep root must be a payload dict.")

    best_points = 1
    best_duration = 0.0

    def walk(payload: Dict[str, Any], points: int, step_time: float) -> None:
        """
        Visit one node and recursively inspect its descendants.

        Args:
            payload (Dict[str, Any]): Current buffered-tree node.
            points (int): Point count inherited from the parent path.
            step_time (float): Point spacing inherited from the parent path.

        Raises:
            TypeError: If the node, ``sweeps``, or ``nodes`` has an invalid
                type.

        """
        nonlocal best_points, best_duration

        if not isinstance(payload, dict):
            raise TypeError("Each buffered sweep node must be a payload dict.")

        sweeps = payload.get("sweeps", []) or []
        if not isinstance(sweeps, (list, tuple)):
            raise TypeError("'sweeps' must be a list or tuple of sweep objects.")

        current_points = points
        current_step_time = step_time
        for sweep in sweeps:
            current_points *= len(sweep.values)
            current_step_time = float(sweep.delay)

        current_duration = current_points * current_step_time
        best_points = max(best_points, current_points)
        best_duration = max(best_duration, current_duration)

        children = payload.get("nodes", []) or []
        if not isinstance(children, list):
            raise TypeError("'nodes' must be a list of child payload dicts.")
        for child in children:
            walk(child, current_points, current_step_time)

    walk(root_payload, points=1, step_time=0.0)
    return best_points, best_duration


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
            trigger_type: Optional[str],
            state: Dict[str, Any],
        ) -> Optional[Dict[str, Any]]

    If the visitor returns a dict, the following keys (if present) are propagated
    down this node's subtree only:
        - "num_points" : int
        - "step_time"  : float
        - "trigger_type": optional str
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
        trigger_type: Optional[str] = None,
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
            trigger_type_in: Optional[str],
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
            curr_trigger_type = payload.get("trigger_type", trigger_type_in)
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
                trigger_type=curr_trigger_type,
                state=curr_state,
            )

            if isinstance(ret, dict):
                curr_num_points = ret.get("num_points", curr_num_points)
                curr_step_time = ret.get("step_time", curr_step_time)
                curr_trigger_type = ret.get("trigger_type", curr_trigger_type)
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
                    curr_trigger_type,
                    curr_state,
                    curr_path,
                )

            return curr_toplevel

        root_name = _name(root_payload, 1, default_root=True)
        return walk_one(
            root_name,
            root_payload,
            None,
            toplevel,
            num_points,
            step_time,
            trigger_type,
            state,
            path,
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
    trigger_type,
    state,
    **kwargs,
):
    """
    Configure instruments and propagate sweep values.

    """

    structural_keys = {"name", "instrument", "dependent", "sweeps", "nodes"}

    if "sweeps" in payload:
        trigger_type = trigger_type or "step"

        sweep_control_keys = {
            "input_trigger",
            "output_trigger",
            "trigger_type",
            "trigger_width",
        }
        sweep_kwargs = {
            key: value
            for key, value in payload.items()
            if key not in structural_keys | sweep_control_keys
        }

        trigger_type, points_new, inner_step = instrument.register_sweep(
            sweep=payload["sweeps"],
            input_trigger=payload.get("input_trigger"),
            output_trigger=payload.get("output_trigger"),
            trigger_type=trigger_type,
            trigger_width=payload.get("trigger_width", 1e-4),
            **sweep_kwargs,
        )
        num_points *= points_new
        step_time = inner_step
        if parent:
            instrument.run_sweep()

    if "dependent" in payload:
        dependent_kwargs = {
            key: value for key, value in payload.items() if key not in structural_keys
        }

        register_signature = signature(instrument.register_dependent)
        if "trigger_type" in register_signature.parameters:
            dependent_kwargs.setdefault("trigger_type", trigger_type or "step")
        else:
            dependent_kwargs.pop("trigger_type", None)

        instrument.register_dependent(
            dependent=payload["dependent"],
            num=num_points,
            delay=step_time,
            **dependent_kwargs,
        )

    state.setdefault("visited_paths", []).append(path)

    return {
        "num_points": num_points,
        "step_time": step_time,
        "trigger_type": trigger_type,
        "state": state,
    }


@parse_bufsweep_tree
def fetch_dependents_tree(
    node, payload, *, parent, path, instrument, state, only_leaves=False, **kwargs
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
        sweep_shape = list(state.get("sweep_shape", []))
        sweeps = list(state.get("sweeps", []))

        for dep in payload.get("dependent", []):
            state["dependent_tree"][dep] = {
                "sweep_shape": sweep_shape,
                "sweeps": sweeps,
            }


@parse_bufsweep_tree
def abort_instruments(node, payload, *, instrument, path, **kwargs):
    """Abort every instrument in a buffered sweep tree."""
    try:
        instrument.abort()
    except Exception as exc:
        logger.warning("[%s] failed to abort instrument: %s", "/".join(path), exc)


@parse_bufsweep_tree
def fetch_results(
    node, payload, *, parent, path, instrument, state, only_leaves=False, **kwargs
):
    """
    Visitor to build a dependent-tree summary.
    """

    children = payload.get("nodes", [])
    is_leaf = not children
    has_fetch = hasattr(instrument, "fetch") and callable(instrument.fetch)
    should_fetch = (
        has_fetch and (is_leaf if only_leaves else True) and ("dependent" in payload)
    )

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
