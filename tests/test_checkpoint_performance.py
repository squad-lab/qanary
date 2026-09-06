"""
Performance guardrails for checkpointing.

These are *complexity* tests, not wall-clock ones. A checkpoint has to stay
proportional to what the sweep just acquired rather than to the size of the
dataset, and the cheapest way to pin that is to count the work rather than time
it -- a threshold in milliseconds passes or fails on how loaded the runner is,
which makes it noise that people learn to ignore.

The property that matters: growing the dataset must not grow the cost of one
checkpoint. Both stores are covered.

  - the in-memory store, which Qimchi Connect snapshots, is written by region
  - the on-disk store is written key by key, skipping unchanged keys

Measured on the reference machine before these were added, at 12 M points
(91 MiB, one checkpoint):

    to_zarr(mode="w"), whole dataset     123.6 ms   over the 0.1 s interval
    region write, rows since last flush   20.1 ms   6.2x faster

If a change makes these tests fail, the checkpoint has gone back to being
O(dataset) and a large measurement will spend more time saving than measuring.

"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
import zarr

from qanary import sweep as sweep_module
from qanary.sweep import (
    _copy_changed_keys,
    _memory_region,
    _stepper,
    _store_cache_key,
    reset_disk_persist_cache,
)


class CountingStore(zarr.MemoryStore):
    """Memory store that records every key written to it."""

    def __init__(self):
        super().__init__()
        self.writes = []

    def __setitem__(self, key, value):
        self.writes.append(key)
        super().__setitem__(key, value)


def dataset(rows: int, cols: int = 64) -> xr.Dataset:
    """A filled grid shaped like a 2D sweep."""
    return xr.Dataset(
        {"signal": (("x", "y"), np.random.rand(rows, cols))},
        coords={"x": np.arange(float(rows)), "y": np.arange(float(cols))},
    )


def seeded(ds: xr.Dataset) -> CountingStore:
    """Seed a store the way ``Measurement.run`` does, then forget those writes."""
    store = CountingStore()
    ds.to_zarr(store=store, mode="w")
    store.writes.clear()
    return store


class ProbeParameter:
    """Minimal settable parameter for driving one real ``_stepper`` point."""

    def __init__(self, name: str):
        self.name = name
        self.value = None

    def __call__(self, value=None):
        if value is not None:
            self.value = value
        return self.value


class ProbeDependent:
    """Dependent returning the row that ``_stepper`` should persist."""

    name = "signal"

    def __init__(self, values):
        self.values = values

    def __call__(self):
        return self.values


def checkpoint_row(ds: xr.Dataset, store: CountingStore, row: int) -> None:
    """Acquire and checkpoint one row through the production ``_stepper`` path."""
    reset_disk_persist_cache()
    sweep_module._MEMORY_FLUSHED_ROWS[id(store)] = row
    sweep_module.last_save = float("inf")
    parameter = ProbeParameter("x")
    dependent = ProbeDependent(ds["signal"].values[row])
    swept = SimpleNamespace(
        parameter=[parameter],
        values=[ds["x"].values[row]],
        start_delay=0.0,
        delay=0.0,
    )

    # A seeded, region-writable store must never fall through to a full write.
    # Patching the production method makes this test fail if `_persist_memory`
    # regresses, even when both paths touch the same number of chunks.
    with patch.object(
        xr.Dataset,
        "to_zarr",
        side_effect=AssertionError("checkpoint attempted a whole-store write"),
    ):
        _stepper(
            dataset=ds,
            data_location="unused",
            depth=1,
            sweeps=[swept],
            independents=[parameter],
            dependents=[dependent],
            sweep_cache=[0.0],
            bar=SimpleNamespace(update=lambda points: None),
            save_interval=float("inf"),
            memory_store=store,
            disk_store=zarr.MemoryStore(),
        )


class TestMemoryCheckpointStaysLocal:
    """The in-memory store backs live snapshots and is written most often."""

    def test_one_row_uses_the_same_number_of_writes_for_each_dataset_height(self):
        """
        A region checkpoint should issue the same number of store writes whether
        the sweep has 64 rows or 1024. This measures operations, not bytes.

        """
        counts = []
        for rows in (64, 256, 1024):
            ds = dataset(rows)
            store = seeded(ds)
            checkpoint_row(ds, store, row=rows // 2)
            counts.append(len(store.writes))

        assert len(set(counts)) == 1, (
            f"checkpoint write count scales with dataset height: {counts} keys "
            "written "
            f"for 64, 256 and 1024 rows"
        )

    def test_a_region_checkpoint_writes_fewer_keys_than_a_full_rewrite(self):
        ds = dataset(512)

        store = seeded(ds)
        checkpoint_row(ds, store, row=256)
        region = len(store.writes)

        store = seeded(ds)
        ds.to_zarr(store=store, mode="w")
        full = len(store.writes)

        assert region < full


class TestDiskCheckpointStaysLocal:
    """The disk store is a mirror; only what changed should reach it."""

    def test_an_unchanged_dataset_writes_nothing(self):
        ds = dataset(128)
        source = zarr.MemoryStore()
        ds.to_zarr(store=source, mode="w")
        target = CountingStore()
        reset_disk_persist_cache()

        _copy_changed_keys(source, target, _store_cache_key(target))
        first = len(target.writes)
        target.writes.clear()
        _copy_changed_keys(source, target, _store_cache_key(target))

        assert first > 0
        assert target.writes == []

    def test_one_changed_chunk_costs_the_same_however_tall_the_dataset(self):
        """One acquired row must not drag the whole store to disk with it."""
        counts = []
        for rows in (64, 256, 1024):
            ds = dataset(rows)
            source = zarr.MemoryStore()
            ds.to_zarr(store=source, mode="w")
            target = CountingStore()
            reset_disk_persist_cache()
            _copy_changed_keys(source, target, _store_cache_key(target))

            # Acquire one more row and re-encode, as a checkpoint would.
            ds["signal"].values[rows // 2, :] = np.random.rand(64)
            ds.to_zarr(store=source, mode="w")
            target.writes.clear()
            _copy_changed_keys(source, target, _store_cache_key(target))
            counts.append(len(target.writes))

        assert len(set(counts)) == 1, (
            f"disk checkpoint scales with dataset height: {counts} keys written "
            f"for 64, 256 and 1024 rows"
        )


class TestRegionSafety:
    """When a region cannot be identified, the whole store must be rewritten."""

    @pytest.mark.parametrize(
        "position, why",
        [
            (None, "caller does not know where the sweep is"),
            ({"y": 0.0}, "position names a dimension that is not the outer one"),
            ({"x": 999.0}, "position is not a coordinate value"),
        ],
    )
    def test_falls_back_to_a_full_write(self, position, why):
        assert _memory_region(dataset(16), position, 0) is None, why

    def test_a_repeated_coordinate_value_is_ambiguous(self):
        """
        ``CircularSweep`` revisits values, so a coordinate can appear twice and
        the row is no longer implied. Rewriting everything is correct; guessing
        a row is not.

        """
        ds = xr.Dataset(
            {"signal": (("x", "y"), np.zeros((4, 2)))},
            coords={"x": np.array([0.0, 1.0, 1.0, 0.0]), "y": np.arange(2.0)},
        )

        assert _memory_region(ds, {"x": 1.0}, 0) is None

    def test_a_known_position_yields_the_rows_since_the_last_flush(self):
        ds = dataset(32)

        assert _memory_region(ds, {"x": 20.0}, 17) == (17, 21, "x")

    def test_a_position_behind_the_flush_point_is_still_covered(self):
        """A sweep that moves backwards must not leave the row unwritten."""
        ds = dataset(32)

        assert _memory_region(ds, {"x": 5.0}, 20) == (5, 6, "x")
