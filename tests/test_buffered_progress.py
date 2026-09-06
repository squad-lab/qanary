"""
Tests for buffered progress accounting.

The topology tests build small buffered trees from lightweight fake sweeps. They
verify that sweep dimensions on the same root-to-leaf path are multiplied, while
concurrent sibling branches are not added together. The reported point count is
therefore the largest buffered coordinate block, and the time estimate is the
duration of the slowest branch.

The timer test advances a deterministic fake clock halfway through a ten-point
block. It checks that the estimator reports five points but reserves the final
point until ``complete()`` confirms that orchestration and result handling have
finished.

The ``_stepper`` integration test replaces the hardware and Xarray objects with
small fakes. A four-point sweep is registered, the fake root is run once, the
child returns a four-value array, and the result is assigned to the fake dataset.
The important assertion is that the progress bar advances by four buffered
points, not by one buffered block as the old implementation did.

"""

from time import sleep
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from qanary.buffered.sweep import _buffered_sweep_progress_info
from qanary.sweep import _BufferedProgress, _stepper


def sweep(points, delay):
    return SimpleNamespace(values=np.arange(points), delay=delay)


class BufferedSweepProgressInfoTests(TestCase):
    def test_counts_all_dimensions_on_a_path(self):
        tree = {
            "instrument": object(),
            "sweeps": [sweep(4, 0.2), sweep(5, 0.2)],
            "nodes": [
                {
                    "instrument": object(),
                    "sweeps": [sweep(3, 0.1)],
                }
            ],
        }

        points, duration = _buffered_sweep_progress_info(tree)

        self.assertEqual(points, 60)
        self.assertAlmostEqual(duration, 6.0)

    def test_uses_largest_concurrent_branch_without_counting_dependents(self):
        tree = {
            "instrument": object(),
            "sweeps": [sweep(10, 0.01)],
            "nodes": [
                {"instrument": object(), "dependent": [object(), object()]},
                {
                    "instrument": object(),
                    "sweeps": [sweep(4, 0.02)],
                    "dependent": [object()],
                },
            ],
        }

        points, duration = _buffered_sweep_progress_info(tree)

        self.assertEqual(points, 40)
        self.assertAlmostEqual(duration, 0.8)

    def test_defaults_to_one_point_without_a_sweep(self):
        tree = {"instrument": object(), "dependent": [object()]}

        self.assertEqual(_buffered_sweep_progress_info(tree), (1, 0.0))

    def test_duration_uses_slowest_concurrent_branch(self):
        tree = {
            "instrument": object(),
            "nodes": [
                {"instrument": object(), "sweeps": [sweep(100, 0.01)]},
                {"instrument": object(), "sweeps": [sweep(10, 0.5)]},
            ],
        }

        self.assertEqual(_buffered_sweep_progress_info(tree), (100, 5.0))


class FakeBar:
    def __init__(self):
        self.n = 0

    def update(self, points):
        self.n += points


class FakeStopEvent:
    def __init__(self):
        self.waits = iter((False, True))

    def wait(self, timeout):
        return next(self.waits)

    def set(self):
        pass


class FakeArrayLocation:
    def __init__(self):
        self.value = None

    def __setitem__(self, indexers, value):
        self.value = value


class FakeArray:
    def __init__(self):
        self.loc = FakeArrayLocation()


class FakeDataset:
    def __init__(self, dependent_name):
        self.data_vars = {dependent_name: FakeArray()}

    def to_zarr(self, *args, **kwargs):
        pass


class FakeDependent:
    name = "signal"


class FakeSweepNode:
    def __init__(self):
        self.toplevel = False
        self.runs = 0

    def register_sweep(self, sweep, **kwargs):
        return "step", len(sweep[0].values), sweep[0].delay

    def run_sweep(self):
        self.runs += 1

    def abort(self):
        pass


class FakeAcquisitionNode:
    def register_dependent(self, dependent, **kwargs):
        self.dependents = dependent

    def fetch(self):
        return [np.arange(4)]

    def abort(self):
        pass


class BufferedStepperProgressTests(TestCase):
    def test_timer_reserves_final_point_until_block_completes(self):
        bar = FakeBar()
        progress = _BufferedProgress(bar, points=10, estimated_duration=1.0)
        progress._stop_event = FakeStopEvent()

        with patch("qanary.sweep.monotonic", side_effect=(0.0, 0.5)):
            progress._run()

        self.assertEqual(bar.n, 5)
        progress.complete()
        self.assertEqual(bar.n, 10)

    def test_pure_buffered_block_advances_by_its_array_length(self):
        root = FakeSweepNode()
        dependent = FakeDependent()
        dataset = FakeDataset(dependent.name)
        bar = FakeBar()
        tree = {
            "instrument": root,
            "sweeps": [sweep(4, 0.0)],
            "nodes": [
                {
                    "instrument": FakeAcquisitionNode(),
                    "dependent": [dependent],
                }
            ],
        }

        result = _stepper(
            dataset=dataset,
            data_location="unused",
            depth=0,
            sweeps=[],
            independents=[],
            dependents=[],
            sweep_cache=[],
            bar=bar,
            buffered_sweep=tree,
        )

        self.assertIs(result, dataset)
        self.assertEqual(root.runs, 1)
        self.assertEqual(bar.n, 4)
        np.testing.assert_array_equal(
            dataset.data_vars[dependent.name].loc.value,
            np.arange(4),
        )


class RecordingBar:
    """Progress bar that records every increment it is given."""

    def __init__(self):
        self.n = 0
        self.increments = []

    def update(self, points):
        self.increments.append(points)
        self.n += points


class ScriptedStopEvent:
    """Stop event that allows a fixed number of ticks, then stops."""

    def __init__(self, ticks):
        self.remaining = ticks
        self.timeouts = []

    def wait(self, timeout):
        self.timeouts.append(timeout)
        if self.remaining <= 0:
            return True
        self.remaining -= 1
        return False

    def set(self):
        self.remaining = 0


class BufferedProgressSmoothnessTests(TestCase):
    """
    The bar must creep forward, not jump.

    A buffered block hands a whole coordinate array to the instrument and gets
    it back in one fetch, so there is no per-point callback to drive the bar.
    ``_BufferedProgress`` interpolates against elapsed time instead, and these
    pin the properties that make that read as smooth progress rather than a
    stalled bar that lurches at the end.

    """

    def _drive(self, points, duration, ticks):
        """
        Run the estimator over a deterministic clock.

        Args:
            points (int): Points in the buffered block.
            duration (float): Estimated block duration in seconds.
            ticks (int): Number of timer wake-ups to simulate.

        Returns:
            tuple[RecordingBar, _BufferedProgress]: Bar and estimator after the
                run, before ``complete()``.

        """
        bar = RecordingBar()
        progress = _BufferedProgress(bar, points=points, estimated_duration=duration)
        progress._stop_event = ScriptedStopEvent(ticks)
        # One clock read to start, then one per tick, spread evenly across the
        # estimated duration.
        step = duration / ticks
        clock = [0.0] + [step * (i + 1) for i in range(ticks)]

        with patch("qanary.sweep.monotonic", side_effect=clock):
            progress._run()

        return bar, progress

    def test_progress_never_moves_backwards(self):
        bar, _ = self._drive(points=40, duration=2.0, ticks=20)

        self.assertTrue(bar.increments, "the bar never moved")
        for increment in bar.increments:
            self.assertGreater(increment, 0)

    def test_progress_never_overshoots_before_completion(self):
        bar, progress = self._drive(points=40, duration=2.0, ticks=40)

        # The final point is reserved for complete(), which signals that results
        # were actually fetched and stored.
        self.assertLessEqual(bar.n, progress.points - 1)

    def test_progress_advances_in_small_steps(self):
        points, duration, ticks = 40, 2.0, 20
        bar, _ = self._drive(points, duration, ticks)

        # Evenly spaced wake-ups over a linear estimate mean each step should be
        # about points/ticks. Allow slack for integer truncation, but a jump of
        # a quarter of the block would read as a lurch.
        expected = points / ticks
        self.assertLessEqual(max(bar.increments), expected + 1)
        self.assertGreaterEqual(len(bar.increments), ticks // 2)

    def test_refresh_interval_keeps_the_bar_alive(self):
        # A slow block must still repaint often enough to look live, and a fast
        # one must not spin. The estimator clamps the interval to [0.01, 0.1].
        cases = ((10, 100.0, 0.1), (10000, 1.0, 0.01), (40, 2.0, 0.05))
        for points, duration, expected in cases:
            with self.subTest(points=points, duration=duration):
                progress = _BufferedProgress(
                    RecordingBar(), points=points, estimated_duration=duration
                )
                stop_event = ScriptedStopEvent(ticks=0)
                progress._stop_event = stop_event

                with patch("qanary.sweep.monotonic", return_value=0.0):
                    progress._run()

                self.assertEqual(len(stop_event.timeouts), 1)
                self.assertAlmostEqual(stop_event.timeouts[0], expected)

    def test_overrunning_block_holds_at_the_final_point(self):
        # The instrument took three times as long as estimated. The bar must sit
        # at points-1 rather than running past the total.
        bar = RecordingBar()
        progress = _BufferedProgress(bar, points=10, estimated_duration=1.0)
        progress._stop_event = ScriptedStopEvent(5)
        clock = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

        with patch("qanary.sweep.monotonic", side_effect=clock):
            progress._run()

        self.assertEqual(bar.n, 9)
        progress.complete()
        self.assertEqual(bar.n, 10)

    def test_complete_lands_exactly_on_the_total(self):
        for ticks in (1, 7, 20, 50):
            bar, progress = self._drive(points=40, duration=2.0, ticks=ticks)
            progress.complete()
            self.assertEqual(bar.n, 40, f"off-total after {ticks} ticks")

    def test_abandoned_block_does_not_reach_the_total(self):
        # stop() is what the failure path calls; only complete() may fill the bar.
        bar, progress = self._drive(points=40, duration=2.0, ticks=10)
        progress.stop()

        self.assertLess(bar.n, 40)

    def test_short_or_instant_blocks_never_start_a_thread(self):
        for points, duration in ((1, 1.0), (10, 0.0)):
            bar = RecordingBar()
            progress = _BufferedProgress(
                bar, points=points, estimated_duration=duration
            )
            progress.start()
            self.assertIsNone(progress._thread)
            progress.complete()
            self.assertEqual(bar.n, progress.points)

    def test_real_timed_block_is_monotonic_and_exact(self):
        # The deterministic tests above pin the arithmetic; this one exercises
        # the actual thread and clock.
        bar = RecordingBar()
        progress = _BufferedProgress(bar, points=20, estimated_duration=0.4)
        progress.start()
        sleep(0.45)
        progress.complete()

        self.assertTrue(all(inc > 0 for inc in bar.increments))
        self.assertGreaterEqual(len(bar.increments), 2, "bar moved in one jump")
        self.assertEqual(bar.n, 20)
