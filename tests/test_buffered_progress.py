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

The stepper integration test replaces the hardware and xarray objects with small
fakes. A four-point sweep is registered, the fake root is run once, the child
returns a four-value array, and the result is assigned to the fake dataset. The
important assertion is that the progress bar advances by four buffered points,
not by one buffered block as the old implementation did.

"""

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from qcutils.buffered.sweep import _buffered_sweep_progress_info
from qcutils.sweep import _BufferedProgress, stepper


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

        with patch("qcutils.sweep.monotonic", side_effect=(0.0, 0.5)):
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

        result = stepper(
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
