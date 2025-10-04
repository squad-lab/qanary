from functools import wraps
from typing import Any, Dict, List, Optional, Tuple


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
