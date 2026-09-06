"""
Tests for the buffered sweep tree walker and its visitors.

A buffered sweep is described as a tree of payload dicts: each node names an
instrument, optionally the sweeps it drives and the dependents it reads, and
optionally child nodes. ``_parse_bufsweep_tree`` is a decorator that turns a
visitor function into a walker over that structure, and the four visitors --
arm, fetch-dependents, abort, fetch-results -- are the whole buffered
orchestration.

Everything here is exercised with fakes. A buffered node is duck-typed
(``register_sweep``, ``register_dependent``, ``run_sweep``, ``fetch``,
``abort``), so the fakes below are a complete stand-in for the real
instrument classes without needing hardware.

The walker's propagation rules carry the physics: point counts multiply down a
root-to-leaf path because nested buffered sweeps span a grid, and a value a
visitor returns applies to that node's subtree only, so a sibling branch is
never given another branch's timing.

"""

from types import SimpleNamespace

import numpy as np
import pytest

from qanary.buffered.sweep import (
    _abort_instruments,
    _arm_instruments,
    _buffered_sweep_progress_info,
    _fetch_dependents_tree,
    _fetch_results,
    _parse_bufsweep_tree,
)


def fake_sweep(points: int, delay: float = 0.0) -> SimpleNamespace:
    """
    Build the minimum a buffered tree needs to look like a sweep.

    Args:
        points (int): Number of coordinate values.
        delay (float): Per-point spacing in seconds.

    Returns:
        SimpleNamespace: Object exposing ``values`` and ``delay``.

    """
    return SimpleNamespace(values=np.arange(points), delay=delay)


class FakeNode:
    """
    A buffered instrument node that records what it was asked to do.

    Attributes:
        calls (list[tuple]): Every call made to it, in order.
        toplevel (bool): Set by the walker on the root instrument.

    """

    def __init__(self, *, points: int = 1, step_time: float = 0.0, fetches=None):
        self.points = points
        self.step_time = step_time
        self.fetches = fetches
        self.calls: list[tuple] = []
        self.toplevel = False

    def register_sweep(self, sweep, **kwargs):
        self.calls.append(("register_sweep", sweep, kwargs))
        return "step", self.points, self.step_time

    def register_dependent(self, dependent, num, delay, **kwargs):
        self.calls.append(("register_dependent", dependent, num, delay, kwargs))

    def run_sweep(self):
        self.calls.append(("run_sweep",))

    def abort(self):
        self.calls.append(("abort",))

    def fetch(self):
        self.calls.append(("fetch",))
        if self.fetches is None:
            raise AssertionError("this node was not meant to be fetched from")
        return self.fetches

    def named(self, call: str) -> list[tuple]:
        """
        Return every recorded call of one kind.

        Args:
            call (str): Method name to filter on.

        Returns:
            list[tuple]: Matching call records.

        """
        return [record for record in self.calls if record[0] == call]


class SweepOnlyNode(FakeNode):
    """A node that sweeps but cannot be read -- no ``fetch``."""

    fetch = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        del self.fetches


class TestWalkerStructure:
    """The schema the tree must follow, and how the walker names its nodes."""

    @pytest.fixture
    def visits(self):
        """
        A visitor that records the walker's arguments at every node.

        Returns:
            tuple[list[dict], Callable]: The record list and the walker.

        """
        seen = []

        @_parse_bufsweep_tree
        def record(node, payload, *, parent, path, instrument, toplevel, **kwargs):
            seen.append(
                {
                    "node": node,
                    "parent": parent,
                    "path": path,
                    "instrument": instrument,
                    "toplevel": toplevel,
                    **kwargs,
                }
            )

        return seen, record

    def test_rejects_a_root_that_is_not_a_payload_dict(self, visits):
        _, walk = visits

        with pytest.raises(TypeError, match="Root must be a payload dict"):
            walk([{"instrument": FakeNode()}])

    def test_rejects_a_root_with_no_recognised_keys(self, visits):
        _, walk = visits

        with pytest.raises(ValueError, match="Root payload must contain"):
            walk({"name": "orphan"})

    def test_rejects_children_that_are_not_a_list(self, visits):
        _, walk = visits

        with pytest.raises(TypeError, match="'nodes' must be a list"):
            walk({"instrument": FakeNode(), "nodes": {"instrument": FakeNode()}})

    def test_rejects_a_child_that_is_not_a_payload_dict(self, visits):
        _, walk = visits

        with pytest.raises(TypeError, match="Each child must be a payload dict"):
            walk({"instrument": FakeNode(), "nodes": ["not a dict"]})

    def test_names_the_root_and_numbers_unnamed_children(self, visits):
        seen, walk = visits

        walk(
            {
                "instrument": FakeNode(),
                "nodes": [
                    {"instrument": FakeNode()},
                    {"instrument": FakeNode()},
                ],
            }
        )

        assert [visit["node"] for visit in seen] == ["root", "node1", "node2"]

    def test_an_explicit_name_wins(self, visits):
        seen, walk = visits

        walk(
            {
                "name": "qdac",
                "instrument": FakeNode(),
                "nodes": [{"name": "dmm", "instrument": FakeNode()}],
            }
        )

        assert [visit["node"] for visit in seen] == ["qdac", "dmm"]

    def test_a_blank_name_falls_back_to_the_default(self, visits):
        seen, walk = visits

        walk({"name": "   ", "instrument": FakeNode()})

        assert seen[0]["node"] == "root"

    def test_path_records_the_route_from_the_root(self, visits):
        seen, walk = visits

        walk(
            {
                "name": "a",
                "instrument": FakeNode(),
                "nodes": [
                    {
                        "name": "b",
                        "instrument": FakeNode(),
                        "nodes": [{"name": "c", "instrument": FakeNode()}],
                    }
                ],
            }
        )

        assert [visit["path"] for visit in seen] == [
            ("a",),
            ("a", "b"),
            ("a", "b", "c"),
        ]
        assert [visit["parent"] for visit in seen] == [None, "a", "b"]

    def test_the_root_instrument_is_marked_and_propagated_as_toplevel(self, visits):
        """
        Only the root actually starts a sweep; every other node arms itself and
        waits for the root's triggers, so they all need to know which node is
        the root.

        """
        seen, walk = visits
        root = FakeNode()
        child = FakeNode()

        returned = walk({"instrument": root, "nodes": [{"instrument": child}]})

        assert root.toplevel is True
        assert child.toplevel is False
        assert all(visit["toplevel"] is root for visit in seen)
        assert returned is root

    def test_a_visitor_return_propagates_down_its_own_subtree_only(self):
        """
        Sibling branches run concurrently, so timing derived in one must not
        leak into the other.

        """
        seen = []

        @_parse_bufsweep_tree
        def record(node, payload, *, num_points, step_time, state, **kwargs):
            seen.append((node, num_points, step_time))
            if node == "left":
                return {"num_points": 100, "step_time": 0.5}

        record(
            {
                "instrument": FakeNode(),
                "nodes": [
                    {
                        "name": "left",
                        "instrument": FakeNode(),
                        "nodes": [{"name": "left_child", "instrument": FakeNode()}],
                    },
                    {"name": "right", "instrument": FakeNode()},
                ],
            }
        )

        assert dict((node, points) for node, points, _ in seen) == {
            "root": 1,
            "left": 1,
            "left_child": 100,
            "right": 1,
        }

    def test_a_visitor_can_replace_the_state_for_its_subtree(self):
        seen = {}

        @_parse_bufsweep_tree
        def record(node, payload, *, state, **kwargs):
            seen[node] = state
            if node == "root":
                return {"state": {"branch": True}}

        record({"instrument": FakeNode(), "nodes": [{"instrument": FakeNode()}]})

        assert seen["node1"] == {"branch": True}

    def test_state_is_shared_between_nodes_by_default(self):
        """
        The collecting visitors rely on this: they accumulate the whole tree's
        dependents into the one dict the caller passed in.

        """

        @_parse_bufsweep_tree
        def collect(node, payload, *, state, **kwargs):
            state.setdefault("nodes", []).append(node)

        state = {}
        collect(
            {"instrument": FakeNode(), "nodes": [{"instrument": FakeNode()}]},
            state=state,
        )

        assert state["nodes"] == ["root", "node1"]

    def test_trigger_type_is_inherited_and_can_be_overridden_per_node(self):
        seen = {}

        @_parse_bufsweep_tree
        def record(node, payload, *, trigger_type, **kwargs):
            seen[node] = trigger_type

        record(
            {
                "instrument": FakeNode(),
                "trigger_type": "start",
                "nodes": [
                    {"name": "inherits", "instrument": FakeNode()},
                    {
                        "name": "overrides",
                        "instrument": FakeNode(),
                        "trigger_type": "step",
                    },
                ],
            }
        )

        assert seen == {"root": "start", "inherits": "start", "overrides": "step"}


class TestArmInstruments:
    """Configuring every instrument in the tree before the block runs."""

    def test_a_sweep_node_is_registered_with_its_control_arguments(self):
        root = FakeNode(points=8, step_time=0.01)
        swept = fake_sweep(8, 0.01)

        _arm_instruments(
            {
                "instrument": root,
                "sweeps": [swept],
                "input_trigger": 2,
                "output_trigger": 3,
                "trigger_width": 5e-4,
            }
        )

        _, sweeps, kwargs = root.named("register_sweep")[0]
        assert sweeps == [swept]
        assert kwargs["input_trigger"] == 2
        assert kwargs["output_trigger"] == 3
        assert kwargs["trigger_width"] == 5e-4
        assert kwargs["trigger_type"] == "step", "the default when none is set"

    def test_trigger_width_defaults_to_a_hundred_microseconds(self):
        root = FakeNode(points=2)

        _arm_instruments({"instrument": root, "sweeps": [fake_sweep(2)]})

        assert root.named("register_sweep")[0][2]["trigger_width"] == 1e-4

    def test_unrecognised_keys_are_forwarded_as_instrument_options(self):
        """
        A node payload doubles as the instrument's configuration, so anything
        that is not structural or trigger control belongs to the driver.

        """
        root = FakeNode(points=2)

        _arm_instruments(
            {
                "name": "mfli",
                "instrument": root,
                "sweeps": [fake_sweep(2)],
                "nodes": [],
                "demod_index": 3,
            }
        )

        kwargs = root.named("register_sweep")[0][2]
        assert kwargs["demod_index"] == 3
        assert not {"name", "instrument", "nodes", "sweeps"} & set(kwargs)

    def test_point_counts_multiply_down_a_path(self):
        """Nested buffered sweeps span a grid, so their points multiply."""
        outer = FakeNode(points=4, step_time=0.1)
        inner = FakeNode(points=5, step_time=0.02)
        reader = FakeNode()

        _arm_instruments(
            {
                "instrument": outer,
                "sweeps": [fake_sweep(4, 0.1)],
                "nodes": [
                    {
                        "instrument": inner,
                        "sweeps": [fake_sweep(5, 0.02)],
                        "nodes": [
                            {"instrument": reader, "dependent": [object()]},
                        ],
                    }
                ],
            }
        )

        _, _, num, delay, _ = reader.named("register_dependent")[0]
        assert num == 20
        assert delay == 0.02, "the innermost sweep sets the point spacing"

    def test_only_child_sweep_nodes_are_started_during_arming(self):
        """
        The root's ``run_sweep`` is the caller's to make, once every other
        instrument is armed. A child sweep is started here because it must
        already be waiting for the root's triggers.

        """
        root = FakeNode(points=3)
        child = FakeNode(points=2)

        _arm_instruments(
            {
                "instrument": root,
                "sweeps": [fake_sweep(3)],
                "nodes": [{"instrument": child, "sweeps": [fake_sweep(2)]}],
            }
        )

        assert root.named("run_sweep") == []
        assert child.named("run_sweep") == [("run_sweep",)]

    def test_a_dependent_is_registered_with_the_accumulated_geometry(self):
        root = FakeNode(points=6, step_time=0.05)
        dependent = object()

        _arm_instruments(
            {
                "instrument": root,
                "sweeps": [fake_sweep(6, 0.05)],
                "dependent": [dependent],
            }
        )

        _, deps, num, delay, _ = root.named("register_dependent")[0]
        assert deps == [dependent]
        assert num == 6
        assert delay == 0.05

    def test_trigger_type_is_dropped_for_drivers_that_do_not_accept_it(self):
        """
        Node classes differ in signature, and passing an unexpected keyword
        would be a TypeError deep inside a run rather than a configuration
        error up front.

        """

        class NoTriggerType(FakeNode):
            def register_dependent(self, dependent, num, delay):
                self.calls.append(("register_dependent", dependent, num, delay, {}))

        node = NoTriggerType()

        _arm_instruments({"instrument": node, "dependent": [object()]})

        assert node.named("register_dependent")[0][4] == {}

    def test_records_every_path_it_visited(self):
        root = FakeNode()
        state = {}

        _arm_instruments(
            {"instrument": root, "nodes": [{"instrument": FakeNode()}]},
            state=state,
        )

        assert state["visited_paths"] == [("root",), ("root", "node1")]


class TestFetchDependentsTree:
    """Collecting the dependents and their geometry before the run starts."""

    def test_collects_each_dependent_with_the_sweeps_above_it(self):
        """
        ``Measurement._make_dataset`` uses this to size the buffered
        dimensions of the dataset, so the sweeps have to be the ones on the
        path to that dependent.

        """
        dependent = object()
        outer = fake_sweep(4)
        inner = fake_sweep(5)
        state = {}

        _fetch_dependents_tree(
            {
                "instrument": FakeNode(),
                "sweeps": [outer],
                "nodes": [
                    {
                        "instrument": FakeNode(),
                        "sweeps": [inner],
                        "dependent": [dependent],
                    }
                ],
            },
            state=state,
        )

        entry = state["dependent_tree"][dependent]
        assert entry["sweep_shape"] == [4, 5]
        assert entry["sweeps"] == [outer, inner]

    def test_a_node_without_fetch_contributes_no_dependents(self):
        state = {}

        _fetch_dependents_tree(
            {"instrument": SweepOnlyNode(), "sweeps": [fake_sweep(3)]},
            state=state,
        )

        assert state["dependent_tree"] == {}

    def test_the_collector_exists_even_for_an_empty_tree(self):
        state = {}

        _fetch_dependents_tree({"instrument": SweepOnlyNode()}, state=state)

        assert state["dependent_tree"] == {}

    def test_a_dependent_carries_only_its_own_branches_sweeps(self):
        """
        Sweep accumulation is branch-local. It used to be shared across the
        whole walk, so a dependent on the second branch was described by both
        branches' sweeps -- harmless for the single-spine trees used in
        practice, but it would have sized a genuinely branching tree's dataset
        wrongly.

        """
        dependent = object()
        state = {}

        _fetch_dependents_tree(
            {
                "instrument": FakeNode(),
                "nodes": [
                    {"instrument": FakeNode(), "sweeps": [fake_sweep(4)]},
                    {
                        "instrument": FakeNode(),
                        "sweeps": [fake_sweep(5)],
                        "dependent": [dependent],
                    },
                ],
            },
            state=state,
        )

        assert state["dependent_tree"][dependent]["sweep_shape"] == [5]


class TestAbortInstruments:
    """Stopping a buffered block, which happens on Ctrl-C mid-run."""

    def test_aborts_every_node_in_the_tree(self):
        nodes = [FakeNode() for _ in range(3)]

        _abort_instruments(
            {
                "instrument": nodes[0],
                "nodes": [
                    {"instrument": nodes[1]},
                    {"instrument": nodes[2]},
                ],
            }
        )

        assert all(node.named("abort") for node in nodes)

    def test_a_node_that_cannot_be_aborted_does_not_stop_the_others(self, caplog):
        """
        Abort runs while the user is already interrupting a run; one wedged
        instrument must not leave the rest armed and triggering.

        """

        class Wedged(FakeNode):
            def abort(self):
                raise TimeoutError("instrument is not responding")

        healthy = FakeNode()

        _abort_instruments(
            {
                "name": "wedged",
                "instrument": Wedged(),
                "nodes": [{"instrument": healthy}],
            }
        )

        assert healthy.named("abort")
        assert "failed to abort instrument" in caplog.text


class TestFetchResults:
    """Reading a completed buffered block back out of the instruments."""

    def test_results_are_reshaped_to_the_sweep_geometry(self):
        """
        Instruments return one flat buffer; the dataset expects the grid the
        buffered sweeps span.

        """
        dependent = object()
        reader = FakeNode(fetches=[np.arange(12.0)])
        state = {}

        _fetch_results(
            {
                "instrument": SweepOnlyNode(),
                "sweeps": [fake_sweep(3), fake_sweep(4)],
                "nodes": [{"instrument": reader, "dependent": [dependent]}],
            },
            state=state,
        )

        result = state["results_tree"][dependent]["result"]
        assert result.shape == (3, 4)
        np.testing.assert_allclose(result, np.arange(12.0).reshape(3, 4))

    def test_a_dependent_with_no_sweeps_keeps_its_flat_result(self):
        dependent = object()
        reader = FakeNode(fetches=[np.arange(5.0)])
        state = {}

        _fetch_results(
            {"instrument": reader, "dependent": [dependent]},
            state=state,
        )

        np.testing.assert_allclose(
            state["results_tree"][dependent]["result"], np.arange(5.0)
        )

    def test_several_dependents_take_the_arrays_in_order(self):
        first, second = object(), object()
        reader = FakeNode(fetches=[np.zeros(3), np.ones(3)])
        state = {}

        _fetch_results(
            {"instrument": reader, "dependent": [first, second]},
            state=state,
        )

        np.testing.assert_allclose(state["results_tree"][first]["result"], np.zeros(3))
        np.testing.assert_allclose(state["results_tree"][second]["result"], np.ones(3))

    def test_a_node_with_no_dependent_is_not_fetched_from(self):
        sweeper_node = FakeNode()
        state = {}

        _fetch_results(
            {"instrument": sweeper_node, "sweeps": [fake_sweep(3)]}, state=state
        )

        assert sweeper_node.named("fetch") == []
        assert state["results_tree"] == {}

    def test_a_failing_fetch_is_logged_and_leaves_no_result(self, caplog):
        """
        A timed-out readout must not take down the other branches' results,
        which are already in hand.

        """

        class Broken(FakeNode):
            def fetch(self):
                raise TimeoutError("no data in the buffer")

        good = object()
        bad = object()
        state = {}

        _fetch_results(
            {
                "instrument": SweepOnlyNode(),
                "nodes": [
                    {"name": "broken", "instrument": Broken(), "dependent": [bad]},
                    {
                        "instrument": FakeNode(fetches=[np.ones(2)]),
                        "dependent": [good],
                    },
                ],
            },
            state=state,
        )

        assert bad not in state["results_tree"]
        assert good in state["results_tree"]
        assert "fetch() failed" in caplog.text


class TestProgressInfoValidation:
    """``_buffered_sweep_progress_info`` refuses a malformed tree up front."""

    def test_rejects_a_root_that_is_not_a_dict(self):
        with pytest.raises(TypeError, match="root must be a payload dict"):
            _buffered_sweep_progress_info([{"instrument": object()}])

    def test_rejects_sweeps_that_are_not_a_sequence(self):
        with pytest.raises(TypeError, match="'sweeps' must be a list or tuple"):
            _buffered_sweep_progress_info(
                {"instrument": object(), "sweeps": fake_sweep(4)}
            )

    def test_rejects_children_that_are_not_a_list(self):
        with pytest.raises(TypeError, match="'nodes' must be a list"):
            _buffered_sweep_progress_info(
                {"instrument": object(), "nodes": {"instrument": object()}}
            )

    def test_rejects_a_child_that_is_not_a_dict(self):
        with pytest.raises(TypeError, match="node must be a payload dict"):
            _buffered_sweep_progress_info(
                {"instrument": object(), "nodes": ["not a dict"]}
            )

    def test_accepts_a_tuple_of_sweeps(self):
        assert _buffered_sweep_progress_info(
            {"instrument": object(), "sweeps": (fake_sweep(4, 0.1),)}
        ) == (4, pytest.approx(0.4))

    def test_treats_an_explicit_none_as_no_sweeps(self):
        assert _buffered_sweep_progress_info(
            {"instrument": object(), "sweeps": None, "nodes": None}
        ) == (1, 0.0)


def test_the_walker_keeps_the_visitor_identity():
    """
    ``@wraps`` on the walker keeps the decorated visitors introspectable,
    which is what makes the names in log messages and tracebacks readable.

    """
    assert _arm_instruments.__name__ == "_arm_instruments"
    assert _fetch_results.__doc__ is not None


def test_a_node_may_carry_neither_sweeps_nor_dependents():
    """A pure grouping node is legal, and must simply be walked through."""
    child = FakeNode(fetches=[np.zeros(2)])
    state = {}

    _fetch_results(
        {
            "instrument": SimpleNamespace(toplevel=False),
            "nodes": [{"instrument": child, "dependent": [object()]}],
        },
        state=state,
    )

    assert len(state["results_tree"]) == 1
