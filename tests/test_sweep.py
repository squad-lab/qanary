"""
Tests for the sweep definitions and the parameter movers in ``qcutils.sweep``.

A ``Sweep`` is pure arithmetic -- it turns a start, a stop and either a step
count or a step size into the array of setpoints a measurement will visit --
but it is the arithmetic that decides how long a run takes and
where its points land, so each conversion is pinned here.

``sweeper`` and ``rampdown`` are the movers that drive parameters without
recording anything. ``rampdown`` is a standalone safety helper for returning
parameters to zero.

"""

from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest

from qcutils.sweep import (
    CircularSweep,
    SegmentedSweep,
    Sweep,
    _sweep_param,
    rampdown,
    sweeper,
)

# CircularSweep rejects delay=0.0 as firmly as it rejects no delay at all, so
# tests that only care about setpoints pass the smallest dwell that gets past
# the check.
TICK = 1e-9


class Recorder:
    """
    A parameter that records what it is set to.

    ``_sweep_param`` calls its parameter directly, so anything callable will
    do -- and unlike a real gate this reports the whole history, which is what
    the ordering assertions need.

    Attributes:
        values (list[float]): Every value set, in order.

    """

    def __init__(self) -> None:
        self.values: list[float] = []

    def __call__(self, value: float) -> None:
        self.values.append(value)


class TestSweep:
    """Setpoint generation, and the two ways of specifying resolution."""

    def test_num_gives_an_inclusive_linear_range(self, gates):
        swept = Sweep(gates.x, 0.0, 1.0, num=5)

        np.testing.assert_allclose(swept.values, [0.0, 0.25, 0.5, 0.75, 1.0])
        assert swept.start == 0.0
        assert swept.stop == 1.0
        assert swept.num == 5

    def test_step_is_converted_to_a_point_count(self, gates):
        """``step`` is a convenience over ``num``, not a separate mode."""
        swept = Sweep(gates.x, 0.0, 1.0, step=0.25)

        assert swept.num == 4
        assert len(swept.values) == 4

    def test_step_counts_a_reversed_range_the_same_way(self, gates):
        """Sweeping down is as ordinary as sweeping up."""
        swept = Sweep(gates.x, 1.0, 0.0, step=0.25)

        assert swept.num == 4
        assert swept.values[0] == pytest.approx(1.0)
        assert swept.values[-1] == pytest.approx(0.0)

    def test_ramprate_is_converted_to_a_per_point_delay(self, gates):
        """
        A ramp rate is volts per second; spreading the span over ``num``
        points makes the dwell time the sweep actually applies.

        """
        swept = Sweep(gates.x, 0.0, 2.0, num=10, ramprate=1.0)

        assert swept.delay == pytest.approx(0.2)

    def test_log_spacing_spans_the_same_endpoints(self, gates):
        swept = Sweep(gates.x, 1.0, 1000.0, num=4, spacing="log")

        np.testing.assert_allclose(swept.values, [1.0, 10.0, 100.0, 1000.0])

    def test_requires_a_step_or_a_point_count(self, gates):
        """
        Without one of them there is nothing to sweep over, and defaulting to
        a guess would silently produce a dataset of the wrong shape.

        """
        with pytest.raises(ValueError, match="step or num"):
            Sweep(gates.x, 0.0, 1.0)

    def test_rejects_an_unknown_spacing(self, gates):
        with pytest.raises(ValueError, match="Invalid spacing"):
            Sweep(gates.x, 1.0, 10.0, num=5, spacing="quadratic")

    def test_a_single_parameter_is_wrapped_in_a_list(self, gates):
        """
        ``_stepper`` iterates ``sweep.parameter`` unconditionally, so the
        one-parameter case has to look like the many-parameter case.

        """
        assert Sweep(gates.x, 0.0, 1.0, num=2).parameter == [gates.x]

    def test_several_parameters_share_one_set_of_setpoints(self, gates):
        swept = Sweep([gates.x, gates.y], 0.0, 1.0, num=3)

        assert swept.parameter == [gates.x, gates.y]
        assert len(swept.values) == 3

    def test_delay_and_start_delay_default_to_zero(self, gates):
        swept = Sweep(gates.x, 0.0, 1.0, num=2)

        assert swept.delay == 0.0
        assert swept.start_delay == 0.0

    def test_start_delay_is_kept_for_the_stepper(self, gates):
        assert Sweep(gates.x, 0.0, 1.0, num=2, start_delay=1.5).start_delay == 1.5


class TestCircularSweep:
    """Up and back down, repeated -- used for hysteresis measurements."""

    def test_values_go_up_then_back_down(self, gates):
        circular = CircularSweep(gates.x, 0.0, 1.0, num=3, delay=TICK)

        np.testing.assert_allclose(circular.values, [0.0, 0.5, 1.0, 1.0, 0.5, 0.0])

    def test_repetitions_tile_the_whole_cycle(self, gates):
        circular = CircularSweep(gates.x, 0.0, 1.0, num=2, delay=TICK, repetitions=3)

        assert len(circular.values) == 2 * 2 * 3
        np.testing.assert_allclose(circular.values[:4], circular.values[4:8])

    def test_requires_a_step_or_a_point_count(self, gates):
        with pytest.raises(ValueError, match="step or num"):
            CircularSweep(gates.x, 0.0, 1.0, delay=0.1)

    @pytest.mark.parametrize("delay", [None, 0.0])
    def test_requires_a_non_zero_delay_or_a_ramprate(self, gates, delay):
        """
        A circular sweep retraces its own path, so a zero dwell would make the
        two directions indistinguishable in time -- the class refuses rather
        than assuming one. Note that ``delay=0.0`` is rejected as firmly as
        omitting it, which an ordinary ``Sweep`` accepts.

        """
        kwargs = {} if delay is None else {"delay": delay}

        with pytest.raises(ValueError, match="delay or ramprate"):
            CircularSweep(gates.x, 0.0, 1.0, num=5, **kwargs)

    def test_ramprate_is_converted_to_a_per_point_delay(self, gates):
        circular = CircularSweep(gates.x, 0.0, 2.0, num=10, ramprate=1.0)

        assert circular.delay == pytest.approx(0.2)

    def test_step_is_converted_to_a_point_count(self, gates):
        circular = CircularSweep(gates.x, 0.0, 1.0, step=0.25, delay=TICK)

        assert circular.num == 4
        assert len(circular.values) == 8


class TestSegmentedSweep:
    """A coarse span with a finer window around a feature."""

    @pytest.fixture
    def segmented(self, gates):
        return SegmentedSweep(
            gates.x, start=0.0, stop=10.0, num=100, center=5.0, center_width=1.0
        )

    def test_spans_the_requested_range(self, segmented):
        assert segmented.values[0] == pytest.approx(0.0)
        assert segmented.values[-1] == pytest.approx(10.0)

    def test_the_centre_window_is_sampled_more_finely(self, segmented):
        """
        The whole point of the class: the mean spacing inside the window must
        be far smaller than outside it, or the feature is under-sampled.

        """
        values = segmented.values
        inside = values[(values > 4.5) & (values < 5.5)]
        outside = values[values < 4.5]

        assert np.mean(np.diff(inside)) < np.mean(np.diff(outside))

    def test_builds_three_segments_around_the_centre(self, segmented):
        assert len(segmented.segments) == 3
        assert segmented.segments[1]["start"] == pytest.approx(4.5)
        assert segmented.segments[1]["stop"] == pytest.approx(5.5)

    def test_a_larger_factor_puts_more_points_in_the_window(self, gates):
        coarse = SegmentedSweep(
            gates.x, 0.0, 10.0, num=100, center=5.0, center_width=1.0, factor=2.0
        )
        fine = SegmentedSweep(
            gates.x, 0.0, 10.0, num=100, center=5.0, center_width=1.0, factor=50.0
        )

        def window_points(sweep):
            return np.sum((sweep.values > 4.5) & (sweep.values < 5.5))

        assert window_points(fine) > window_points(coarse)

    def test_exposes_the_dwell_attributes_the_stepper_reads(self, segmented):
        """
        ``_stepper`` reads ``delay`` and ``start_delay`` off every sweep it is
        given, so a segmented sweep has to carry them even though it does not
        use them.

        """
        assert segmented.delay == 0.0
        assert segmented.start_delay == 0.0


class TestSweepParam:
    """The single-sweep mover underneath ``sweeper``."""

    def test_visits_every_setpoint_in_order(self):
        """
        ``sweep.parameter`` is read as one callable here, so this is the shape
        ``sweeper`` produces when it hands over a bare parameter rather than a
        list of them.

        """
        recorder = Recorder()
        swept = SimpleNamespace(
            parameter=recorder,
            values=[0.0, 0.5, 1.0],
            delay=0.0,
        )

        _sweep_param(swept)

        assert recorder.values == [0.0, 0.5, 1.0]

    def test_override_takes_a_parameter_values_and_delay_directly(self):
        """
        The override exists so ``sweeper`` can drive each parameter of a
        multi-parameter sweep on its own thread, without building a Sweep per
        parameter.

        """
        recorder = Recorder()

        _sweep_param(override=[recorder, [1.0, 2.0, 3.0], 0.0])

        assert recorder.values == [1.0, 2.0, 3.0]

    def test_applies_the_delay_between_points(self):
        recorder = Recorder()
        started = monotonic()

        _sweep_param(override=[recorder, [0.0] * 4, 0.01])

        assert monotonic() - started >= 0.04


class TestSweeper:
    """Moving parameters without recording anything."""

    def test_drives_a_single_sweep_to_its_endpoint(self, gates):
        sweeper(Sweep(gates.x, 0.0, 1.0, num=5))

        assert gates.x() == pytest.approx(1.0)

    def test_drives_every_parameter_of_a_multi_parameter_sweep(self, gates):
        sweeper(Sweep([gates.x, gates.y], 0.0, 1.0, num=5))

        assert gates.x() == pytest.approx(1.0)
        assert gates.y() == pytest.approx(1.0)

    def test_drives_a_list_of_sweeps_in_order(self, gates):
        sweeper(
            [
                Sweep(gates.x, 0.0, 1.0, num=3),
                Sweep(gates.y, 0.0, 2.0, num=3),
            ]
        )

        assert gates.x() == pytest.approx(1.0)
        assert gates.y() == pytest.approx(2.0)

    @pytest.mark.parametrize("wrap", [lambda s: s, lambda s: [s]], ids=["bare", "list"])
    def test_drives_a_circular_sweep(self, gates, wrap):
        sweeper(wrap(CircularSweep(gates.x, 0.0, 1.0, num=5, delay=TICK)))

        assert gates.x() == pytest.approx(0.0)

    def test_parallel_sweeps_still_reach_their_endpoints(self, gates):
        """
        The parallel path hands each parameter to its own thread. The threads
        are not joined, so the assertion polls rather than reading once.

        """
        sweeper([Sweep([gates.x, gates.y], 0.0, 1.0, num=20)], parallel=True)

        deadline = monotonic() + 5.0
        while monotonic() < deadline:
            if gates.x() == pytest.approx(1.0) and gates.y() == pytest.approx(1.0):
                break
        assert gates.x() == pytest.approx(1.0)
        assert gates.y() == pytest.approx(1.0)


class TestRampdown:
    """
    The safety path: whatever the parameter reads, bring it to zero.

    A rampdown is always 500 points, and the default dwell is 10 ms, so a real
    one takes five seconds. These tests replace ``sleep`` with a recorder --
    which both keeps the suite fast and lets them assert the dwell directly
    instead of inferring it from a stopwatch.

    """

    @pytest.fixture
    def dwells(self, monkeypatch):
        """
        Capture every dwell ``rampdown`` asks for, without waiting them out.

        Returns:
            list[float]: Populated as the rampdown runs.

        """
        recorded: list[float] = []
        monkeypatch.setattr("qcutils.sweep.sleep", recorded.append)
        return recorded

    def test_brings_a_single_parameter_to_zero(self, gates, dwells):
        gates.x(1.0)

        rampdown(gates.x)

        assert gates.x() == 0.0

    def test_brings_every_parameter_of_a_list_to_zero(self, gates, dwells):
        gates.x(1.0)
        gates.y(-2.0)

        rampdown([gates.x, gates.y])

        assert gates.x() == 0.0
        assert gates.y() == 0.0

    def test_ramps_down_in_five_hundred_steps(self, gates, dwells):
        gates.x(1.0)

        rampdown(gates.x)

        assert len(dwells) == 500

    def test_defaults_to_a_ten_millisecond_dwell(self, gates, dwells):
        """Without a ramp rate there is no safe speed to compute, so it crawls."""
        gates.x(1.0)

        rampdown(gates.x)

        assert set(dwells) == {1e-2}

    def test_ramprate_sets_the_dwell_from_the_excursion(self, gates, dwells):
        """A 1 V excursion at 500 V/s over 500 points is 4 us per point."""
        gates.x(1.0)

        rampdown(gates.x, ramprate=500.0)

        assert dwells == pytest.approx([1.0 / (500.0 * 500)] * 500)

    def test_ramprate_uses_the_largest_excursion_of_the_group(self, gates, dwells):
        """
        One delay drives every parameter, so it has to be the one the furthest
        parameter needs -- otherwise the largest excursion ramps too fast.

        """
        gates.x(0.5)
        gates.y(5.0)

        rampdown([gates.x, gates.y], ramprate=500.0)

        assert dwells == pytest.approx([5.0 / (500.0 * 500)] * 1000)
        assert gates.x() == 0.0
        assert gates.y() == 0.0

    def test_a_parameter_already_at_zero_stays_there(self, gates, dwells):
        rampdown(gates.x, ramprate=500.0)

        assert gates.x() == 0.0
