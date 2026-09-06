"""
Tests for the buffered instrument nodes.

``qcutils.buffered.instruments`` is deprecated and on its way to the
``drivers`` package, but it is still what every buffered measurement script
imports, so the behaviour it has now is worth pinning before it moves.

Scope: this module covers the node logic that is instrument-independent --
sweep and dependent bookkeeping in ``BufferedNodeBase``, the argument
validation each node does before it touches a wire, the geometry it programs
into the instrument, and the shape of what ``fetch`` returns. It does not
cover the ZI (``NodeMFLI``, ``NodeUHFLI``) or Basel DAC nodes' register paths:
those are long sequences of driver-protocol calls, and a fake instrument
faithful enough to test them against would only restate the implementation.
Those paths need hardware.

TODO: The intended DMM 2D count configuration is retained as a strict expected
failure until it can be validated on hardware.

"""

import importlib
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from qcutils.buffered import instruments as instruments_module
from qcutils.buffered.instruments import (
    BufferedNodeBase,
    NodeDelay,
    NodeKeysightDMM,
    NodeKeysightVNA,
    NodeQDAC2,
)
from qcutils.sweep import Sweep


class FakeParam:
    """
    A settable value that records what it was given.

    Instrument settings on these drivers are all of the form
    ``core.trigger.count(5)``, and reading one back is the same call with no
    argument, so one recording callable stands in for any of them.

    Attributes:
        calls (list): Every value set, in order.

    """

    def __init__(self, value=None):
        self.value = value
        self.calls = []

    def __call__(self, *args):
        if not args:
            return self.value
        self.value = args[0]
        self.calls.append(args[0])


class FakeAction:
    """
    A recording stand-in for an instrument command with no settable value.

    ``reset()``, ``abort_measurement()`` and the sweep handle's ``close()``
    are all called for their effect, so what matters is how many times and
    with what -- not a value to read back.

    Attributes:
        calls (list[tuple]): Positional arguments of every call, in order.

    """

    def __init__(self, result=None):
        self.result = result
        self.calls: list[tuple] = []

    def __call__(self, *args, **kwargs):
        self.calls.append(args)
        return self.result


class TestDeprecation:
    """The module warns on import; nothing inside qcutils imports it."""

    def test_importing_warns_that_the_module_is_moving(self):
        with pytest.warns(DeprecationWarning, match="drivers"):
            importlib.reload(instruments_module)

    def test_the_node_classes_are_the_public_api(self):
        assert instruments_module.__all__ == [
            "BufferedNodeBase",
            "NodeMFLI",
            "NodeUHFLI",
            "NodeKeysightDMM",
            "NodeQDAC2",
            "NodeKeysightVNA",
            "NodeBaselDAC",
            "NodeDelay",
        ]

    def test_qcutils_itself_never_imports_it(self):
        """
        The deprecation warning is aimed at measurement scripts. If anything
        inside qcutils imported this module, every user would see the warning
        on ``import qcutils`` with nothing they could do about it.

        """
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-W",
                "error::DeprecationWarning",
                "-c",
                "import qcutils, qcutils.measure, qcutils.sweep, qcutils.dataset",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr


class TestBufferedNodeBase:
    """Sweep and dependent bookkeeping shared by every node."""

    @pytest.fixture
    def node(self):
        return BufferedNodeBase(inst=SimpleNamespace(name="core"))

    def test_a_fresh_node_is_a_leaf_that_can_sweep(self, node):
        assert node.buffered is True
        assert node.toplevel is False
        assert node.endnode is True
        assert node.sweepnode is True

    def test_a_single_sweep_is_wrapped_in_a_list(self, node, gates):
        swept = Sweep(gates.x, 0.0, 1.0, num=5, delay=0.01)

        node._process_sweeps(swept)

        assert node.sweeps == [swept]
        assert node.dims == 1

    def test_registering_a_sweep_makes_the_node_an_interior_node(self, node, gates):
        node._process_sweeps(Sweep(gates.x, 0.0, 1.0, num=5, delay=0.01))

        assert node.endnode is False

    def test_two_sweeps_multiply_into_the_total_point_count(self, node, gates):
        """
        A 2D buffered sweep spans a grid, and ``num`` is what the acquisition
        node is told to expect back.

        """
        node._process_sweeps(
            [
                Sweep(gates.x, 0.0, 1.0, num=4, delay=0.01),
                Sweep(gates.y, 0.0, 1.0, num=5, delay=0.01),
            ]
        )

        assert node.num == 20
        assert node.num_tup == [4, 5]
        assert node.dims == 2

    def test_the_dwell_and_start_delays_are_kept(self, node, gates):
        node._process_sweeps(
            Sweep(gates.x, 0.0, 1.0, num=4, delay=0.01, start_delay=0.5)
        )

        assert node.delay == 0.01
        assert node.start_delay == [0.5]

    def test_sweeps_must_come_from_one_instrument(self, node, gates, instrument):
        """
        A buffered node programs one device; sweeps on two devices cannot be
        driven from a single armed sequence.

        """
        other = instrument("z")

        with pytest.raises(AssertionError, match="same instrument"):
            node._process_sweeps(
                [
                    Sweep(gates.x, 0.0, 1.0, num=4, delay=0.01),
                    Sweep(other.z, 0.0, 1.0, num=4, delay=0.01),
                ]
            )

    def test_sweeps_must_share_a_dwell(self, node, gates):
        """
        The instrument steps on one timebase, so two different dwells cannot
        both be honoured.

        """
        with pytest.raises(AssertionError, match="same delay"):
            node._process_sweeps(
                [
                    Sweep(gates.x, 0.0, 1.0, num=4, delay=0.01),
                    Sweep(gates.y, 0.0, 1.0, num=4, delay=0.02),
                ]
            )

    def test_at_most_two_dimensions_are_supported(self, node, instrument):
        inst = instrument("a", "b", "c")

        with pytest.raises(AssertionError, match="Maximum 2D"):
            node._process_sweeps(
                [
                    Sweep(inst.a, 0.0, 1.0, num=2, delay=0.01),
                    Sweep(inst.b, 0.0, 1.0, num=2, delay=0.01),
                    Sweep(inst.c, 0.0, 1.0, num=2, delay=0.01),
                ]
            )

    def test_a_single_dependent_is_wrapped_in_a_list(self, node, gates):
        node._process_dependents(gates.x)

        assert node.dependents == [gates.x]

    def test_several_dependents_are_kept_in_order(self, node, gates):
        node._process_dependents([gates.x, gates.y])

        assert node.dependents == [gates.x, gates.y]

    def test_aborting_a_node_with_nothing_running_is_a_no_op(self, node):
        assert node.abort() is None


class TestNodeDelay:
    """A virtual sweep node: it defines a timebase and steps nothing."""

    @pytest.fixture
    def node(self):
        return NodeDelay(inst=SimpleNamespace(name="delay"))

    def test_registering_returns_the_point_count_and_spacing(self, node, gates):
        result = node.register_sweep(Sweep(gates.x, 0.0, 1.0, num=10, delay=0.05))

        assert result == (None, 10, 0.05)

    def test_only_one_dimension_is_supported(self, node, gates):
        with pytest.raises(ValueError, match="only supports 1D"):
            node.register_sweep(
                [
                    Sweep(gates.x, 0.0, 1.0, num=4, delay=0.01),
                    Sweep(gates.y, 0.0, 1.0, num=4, delay=0.01),
                ]
            )

    def test_as_the_root_it_holds_the_acquisition_window_open(
        self, node, gates, monkeypatch
    ):
        """
        Nothing is stepped, but the downstream acquisition module needs the
        block to last as long as the sweep it was armed for.

        """
        slept = []
        monkeypatch.setattr("qcutils.buffered.instruments.sleep", slept.append)
        node.register_sweep(Sweep(gates.x, 0.0, 1.0, num=10, delay=0.05))
        node.toplevel = True

        node.run_sweep()

        assert slept == [pytest.approx(11 * 0.05)]

    def test_below_the_root_it_does_nothing(self, node, gates, monkeypatch):
        slept = []
        monkeypatch.setattr("qcutils.buffered.instruments.sleep", slept.append)
        node.register_sweep(Sweep(gates.x, 0.0, 1.0, num=10, delay=0.05))

        node.run_sweep()

        assert slept == []


class TestNodeKeysightVNA:
    """A frequency sweep the VNA runs internally."""

    @pytest.fixture
    def core(self):
        return SimpleNamespace(
            frequency=FakeParam(),
            configure_active_s_parameters=FakeAction(),
            traces=[SimpleNamespace(run_sweep=FakeAction())],
        )

    @pytest.fixture
    def node(self, core):
        return NodeKeysightVNA(inst=core)

    def test_the_sweep_setpoints_are_pushed_to_the_instrument(self, node, core, gates):
        swept = Sweep(gates.x, 1e9, 2e9, num=201, delay=0.0)

        trigger_type, points, step_time = node.register_sweep(swept)

        np.testing.assert_allclose(core.frequency.calls[0], swept.values)
        assert (trigger_type, points, step_time) == (None, 201, None)

    def test_only_one_dimension_is_supported(self, node, gates):
        with pytest.raises(NotImplementedError, match="Only one sweep"):
            node.register_sweep(
                [
                    Sweep(gates.x, 1e9, 2e9, num=11, delay=0.0),
                    Sweep(gates.y, 0.0, 1.0, num=11, delay=0.0),
                ]
            )

    def test_dependents_are_configured_by_their_s_parameter(self, node, core):
        dependents = [SimpleNamespace(_sparam="S21"), SimpleNamespace(_sparam="S11")]

        node.register_dependent(dependents, num=201, delay=0.0)

        assert core.configure_active_s_parameters.calls == [(["S21", "S11"],)]

    def test_running_the_sweep_goes_through_the_first_trace(self, node, core):
        node.run_sweep()

        assert core.traces[0].run_sweep.calls == [()]

    def test_fetching_reads_each_dependent(self, node):
        node._process_dependents([lambda: np.arange(3), lambda: np.arange(3) * 2])

        results = node.fetch()

        np.testing.assert_allclose(results[0], [0, 1, 2])
        np.testing.assert_allclose(results[1], [0, 2, 4])


class FakeDMM:
    """
    A Keysight 344xxA stand-in exposing only what the node touches.

    Attributes:
        fetched (list[float]): What ``fetch`` returns.

    """

    def __init__(self, model: str = "34461A", samples: int = 4):
        self.model = model
        self.trigger = SimpleNamespace(
            source=FakeParam(),
            slope=FakeParam(),
            count=FakeParam(),
            delay=FakeParam(),
        )
        self.sample = SimpleNamespace(count=FakeParam())
        self.autorange = FakeParam()
        self.autozero = FakeParam()
        self.NPLC = FakeParam()
        self.NPLC_list = [0.02, 0.2, 1, 10, 100]
        self.line_frequency = FakeParam(50)
        self.reset = FakeAction()
        self.init_measurement = FakeAction()
        self.device_clear = FakeAction()
        self.abort_measurement = FakeAction()
        self.fetched = list(range(samples))
        self.timeout = SimpleNamespace(set_to=self._timeout_context)
        self.timeouts = []

    def _timeout_context(self, seconds):
        self.timeouts.append(seconds)

        class Scoped:
            def __enter__(inner):
                return None

            def __exit__(inner, *exc):
                return False

        return Scoped()

    def fetch(self):
        return self.fetched


class TestNodeKeysightDMM:
    """Externally triggered buffered acquisition into the DMM's memory."""

    @pytest.fixture
    def core(self):
        return FakeDMM()

    @pytest.fixture
    def node(self, core, monkeypatch):
        monkeypatch.setattr("qcutils.buffered.instruments.sleep", lambda s: None)
        return NodeKeysightDMM(inst=core)

    def test_construction_parks_the_dmm_on_immediate_triggering(self, node, core):
        """So an ordinary scalar read works before a buffered block is armed."""
        assert core.trigger.source.calls == ["IMM"]
        assert core.reset.calls == [()]

    def test_registering_switches_to_external_triggering(self, node, core):
        node.register_dependent([object()], num=10, delay=0.01)

        assert core.trigger.source.calls[-1] == "EXT"
        assert core.trigger.slope.calls == ["POS"]

    def test_autorange_and_autozero_are_disabled(self, node, core):
        """
        Both would insert an unpredictable extra delay between triggers, which
        a buffered block has no room for.

        """
        node.register_dependent([object()], num=10, delay=0.01)

        assert core.autorange.calls == ["OFF"]
        assert core.autozero.calls == ["OFF"]

    def test_a_one_dimensional_block_takes_one_sample_per_trigger(self, node, core):
        node.register_dependent([object()], num=10, delay=0.01)

        assert core.sample.count.calls == [1]
        assert core.trigger.count.calls == [10]

    @pytest.mark.xfail(
        strict=True,
        reason="2D count setup is deferred until it can be verified on the DMM",
    )
    def test_a_two_dimensional_block_splits_samples_from_triggers(self, node, core):
        """
        Record the target contract for ``num=(rows, cols)``.

        The DMM should acquire ``cols`` readings per trigger for ``rows``
        triggers. This remains an expected failure until the configuration is
        validated and implemented on hardware.

        """
        node.register_dependent([object()], num=(8, 3), delay=0.01)

        assert core.sample.count.calls == [3]
        assert core.trigger.count.calls == [8]

    def test_the_trigger_delay_is_a_fifth_of_the_step_time(self, node, core):
        """
        Set last, and unconditionally -- the model-specific delay computed in
        the 1D branch above it is overwritten by this line.

        """
        node.register_dependent([object()], num=10, delay=0.05)

        assert core.trigger.delay.calls[-1] == pytest.approx(0.01)

    def test_the_integration_time_is_the_longest_that_fits_the_step(self, node, core):
        """
        Longer integration is less noise, so the node takes the largest NPLC
        whose duration still fits inside the step's remaining budget.

        """
        node.register_dependent([object()], num=10, delay=1.0)

        # 0.64 s of budget at 50 Hz line frequency admits 10 PLC (0.2 s) but
        # not 100 PLC (2 s).
        assert core.NPLC.calls == [10]

    def test_an_unknown_model_is_rejected(self, monkeypatch):
        monkeypatch.setattr("qcutils.buffered.instruments.sleep", lambda s: None)
        node = NodeKeysightDMM(inst=FakeDMM(model="34465A"))

        with pytest.raises(NotImplementedError, match="34465A"):
            node.register_dependent([object()], num=10, delay=0.01)

    def test_only_the_external_trigger_input_is_supported(self, node):
        with pytest.raises(NotImplementedError, match="input_trigger=1"):
            node.register_dependent([object()], num=10, delay=0.01, input_trigger=2)

    def test_only_step_triggering_is_supported(self, node):
        with pytest.raises(NotImplementedError, match="'ramp' not implemented"):
            node.register_dependent([object()], num=10, delay=0.01, trigger_type="ramp")

    def test_fetching_returns_one_flat_array(self, node, core):
        node.register_dependent([object()], num=4, delay=0.01)

        result = node.fetch()

        assert len(result) == 1
        np.testing.assert_allclose(result[0], [0, 1, 2, 3])

    def test_fetching_sizes_its_timeout_from_the_block(self, node, core):
        """
        The default VISA timeout is far shorter than a buffered block, so it
        is widened to the block's own duration plus a margin.

        """
        node.register_dependent([object()], num=100, delay=1.0)
        node.fetch()

        assert core.timeouts == [pytest.approx(102.0)]

    def test_a_two_dimensional_block_counts_rows_times_columns(self, node, core):
        node.register_dependent([object()], num=(8, 3), delay=1.0)
        node.fetch()

        assert core.timeouts == [pytest.approx(26.0)]

    def test_fetching_always_aborts_afterwards(self, node, core):
        """
        The DMM is left armed for external triggers otherwise, and the next
        ordinary read would block until something triggered it.

        """
        node.register_dependent([object()], num=4, delay=0.01)
        node.fetch()

        assert core.trigger.source.calls[-1] == "IMM"
        assert core.sample.count.calls[-1] == 1
        assert core.trigger.count.calls[-1] == 1

    def test_a_failing_fetch_still_aborts(self, node, core):
        node.register_dependent([object()], num=4, delay=0.01)
        core.fetch = lambda: (_ for _ in ()).throw(TimeoutError("no data"))

        with pytest.raises(TimeoutError):
            node.fetch()

        assert core.trigger.source.calls[-1] == "IMM"


class FakeQDAC:
    """A QDAC2 stand-in covering the arrangement and trigger plumbing."""

    def __init__(self):
        self.free_all_triggers = FakeAction()
        self.external_triggers = [SimpleNamespace(width_s=FakeParam())]
        self.arrangements = []

    def arrange(self, contacts, output_triggers):
        arrangement = SimpleNamespace(
            contacts=contacts,
            output_triggers=output_triggers,
            virtual_sweep=self._sweep,
            virtual_sweep2d=self._sweep,
            get_trigger_by_name=lambda name: name,
        )
        self.arrangements.append(arrangement)
        return arrangement

    def _sweep(self, **kwargs):
        return SimpleNamespace(
            kwargs=kwargs, start=FakeAction(), close=FakeAction(), closed=False
        )


@pytest.fixture
def qdac_gate(instrument):
    """
    A gate whose instrument carries the channel number the QDAC node reads.

    Returns:
        Parameter: A gate on a dummy instrument with ``_channum`` set.

    """
    inst = instrument("ch01")
    inst._channum = 1
    return inst.ch01


class TestNodeQDAC2:
    """Argument validation and trigger routing for the QDAC2 sweep node."""

    @pytest.fixture
    def core(self):
        return FakeQDAC()

    @pytest.fixture
    def node(self, core):
        return NodeQDAC2(inst=core)

    def test_a_one_dimensional_sweep_is_arranged_by_channel(
        self, node, core, qdac_gate
    ):
        trigger_type, points, delay = node.register_sweep(
            Sweep(qdac_gate, 0.0, 1.0, num=10, delay=0.01),
            output_trigger=2,
            trigger_type="step",
        )

        assert node.contacts == {"ch01": 1}
        assert (trigger_type, points, delay) == ("step", 10, 0.01)

    def test_trigger_names_follow_the_qdac_convention(self, node, core, qdac_gate):
        node.register_sweep(
            Sweep(qdac_gate, 0.0, 1.0, num=10, delay=0.01),
            input_trigger=3,
            output_trigger=4,
            trigger_type="step",
        )

        assert node.input_trigger_key == "trigin_3"
        assert node.output_trigger_key == "trigout_4"
        assert node.output_trigger == {"trigout_4": 4}

    def test_absent_triggers_stay_none(self, node, core, qdac_gate):
        node.register_sweep(
            Sweep(qdac_gate, 0.0, 1.0, num=10, delay=0.01), trigger_type="step"
        )

        assert node.input_trigger_key is None
        assert node.output_trigger is None

    def test_the_trigger_width_is_pushed_to_every_external_trigger(
        self, node, core, qdac_gate
    ):
        node.register_sweep(
            Sweep(qdac_gate, 0.0, 1.0, num=10, delay=0.01),
            trigger_type="step",
            trigger_width=5e-4,
        )

        assert core.external_triggers[0].width_s.calls == [5e-4]

    def test_a_dwell_shorter_than_the_trigger_is_rejected(self, node, qdac_gate):
        """
        The trigger pulse would outlast the step it marks, so the downstream
        instrument could not tell two steps apart.

        """
        with pytest.raises(ValueError, match="less than trigger width"):
            node.register_sweep(
                Sweep(qdac_gate, 0.0, 1.0, num=10, delay=1e-5),
                trigger_width=1e-4,
                trigger_type="step",
            )

    def test_an_unknown_trigger_type_is_rejected(self, node, qdac_gate):
        with pytest.raises(ValueError, match='"ramp" or "step"'):
            node.register_sweep(
                Sweep(qdac_gate, 0.0, 1.0, num=10, delay=0.01),
                trigger_type="edge",
            )

    def test_ramp_triggering_makes_no_sense_for_a_one_dimensional_sweep(
        self, node, qdac_gate
    ):
        """A 1D sweep is one ramp, so a per-ramp trigger fires once."""
        with pytest.raises(NotImplementedError, match="'ramp' not implemented"):
            node.register_sweep(Sweep(qdac_gate, 0.0, 1.0, num=10, delay=0.01))

    def test_a_two_dimensional_ramp_sweep_reports_its_shape(
        self, node, core, instrument
    ):
        """
        Both sweeps must be on one instrument -- a QDAC drives both contacts.
        The node then reads a channel number per swept parameter, so a real
        QDAC gives two; the dummy has one, which the shape does not depend on.

        """
        qdac = instrument("outer", "inner")
        qdac._channum = 1
        sweeps = [
            Sweep(qdac.outer, 0.0, 1.0, num=4, delay=0.01),
            Sweep(qdac.inner, 0.0, 1.0, num=5, delay=0.01),
        ]

        trigger_type, points, delay = node.register_sweep(sweeps)

        assert trigger_type == "ramp"
        assert points == [4, 5]
        assert set(node.contacts) == {"outer", "inner"}

    def test_aborting_before_a_sweep_started_is_a_no_op(self, node):
        assert node.abort() is None

    def test_aborting_closes_the_running_sweep_once(
        self, node, core, qdac_gate, monkeypatch
    ):
        monkeypatch.setattr("qcutils.buffered.instruments.sleep", lambda s: None)
        node.register_sweep(
            Sweep(qdac_gate, 0.0, 1.0, num=10, delay=0.01), trigger_type="step"
        )
        node.run_sweep()

        node.abort()
        node.abort()

        assert len(node._qdac_sweep.close.calls) == 1

    def test_the_root_node_holds_the_window_open_then_stops_the_sweep(
        self, node, core, qdac_gate, monkeypatch
    ):
        slept = []
        monkeypatch.setattr("qcutils.buffered.instruments.sleep", slept.append)
        node.register_sweep(
            Sweep(qdac_gate, 0.0, 1.0, num=10, delay=0.01), trigger_type="step"
        )
        node.toplevel = True

        node.run_sweep()

        assert slept == [pytest.approx(11 * 0.01)]
        assert node._sweep_active is False

    def test_fetching_flattens_each_dependent(self, node):
        instrument_stub = SimpleNamespace(
            fetch_current_A=lambda: [[1.0, 2.0], [3.0, 4.0]]
        )
        node._process_dependents(SimpleNamespace(instrument=instrument_stub))

        results = node.fetch()

        np.testing.assert_allclose(results[0], [1.0, 2.0, 3.0, 4.0])


def test_the_module_can_be_imported_without_a_warning_filter():
    """
    A DeprecationWarning is hidden by default in scripts, so importing must
    not fail under ``-W error`` only because of its own warning being escalated
    in an unrelated test.

    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        importlib.reload(instruments_module)
