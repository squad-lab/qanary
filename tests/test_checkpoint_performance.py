"""Checkpoint complexity tests for memory and disk stores."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
import zarr

from qanary import sweep as sweep_module
from qanary.sweep import (
    ZARR_FORMAT,
    _copy_changed_keys,
    _memory_region,
    _stepper,
    _store_cache_key,
    reset_disk_persist_cache,
)


class CountingDict(dict):
    def __init__(self):
        super().__init__()
        self.writes = []

    def __setitem__(self, key, value):
        self.writes.append(key)
        super().__setitem__(key, value)


class CountingStore(zarr.storage.MemoryStore):
    def __init__(self):
        super().__init__()
        self.writes = []

    async def set(self, key, value, *args, **kwargs):
        self.writes.append(key)
        await super().set(key, value, *args, **kwargs)


def dataset(rows: int, cols: int = 64) -> xr.Dataset:
    """A filled grid shaped like a 2D sweep."""
    return xr.Dataset(
        {"signal": (("x", "y"), np.random.rand(rows, cols))},
        coords={"x": np.arange(float(rows)), "y": np.arange(float(cols))},
    )


def seeded(ds: xr.Dataset) -> CountingDict:
    """Seed a store the way ``Measurement.run`` does, then forget those writes."""
    store = CountingDict()
    ds.to_zarr(store=store, mode="w", zarr_format=ZARR_FORMAT)
    store.writes.clear()
    return store


class ProbeParameter:
    def __init__(self, name: str):
        self.name = name
        self.value = None

    def __call__(self, value=None):
        if value is not None:
            self.value = value
        return self.value


class ProbeDependent:
    name = "signal"

    def __init__(self, values):
        self.values = values

    def __call__(self):
        return self.values


def checkpoint_row(ds: xr.Dataset, store: CountingDict, row: int) -> None:
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

    # Fail immediately if a region checkpoint falls back to a full write.
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
            disk_store=zarr.storage.MemoryStore(),
        )


class TestMemoryCheckpointStaysLocal:
    def test_one_row_uses_the_same_number_of_writes_for_each_dataset_height(self):
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
        ds.to_zarr(store=store, mode="w", zarr_format=ZARR_FORMAT)
        full = len(store.writes)

        assert region < full


class TestDiskCheckpointStaysLocal:
    def test_an_unchanged_dataset_writes_nothing(self):
        ds = dataset(128)
        source = {}
        ds.to_zarr(store=source, mode="w", zarr_format=ZARR_FORMAT)
        target = CountingStore()
        reset_disk_persist_cache()

        _copy_changed_keys(source, target, _store_cache_key(target))
        first = len(target.writes)
        target.writes.clear()
        _copy_changed_keys(source, target, _store_cache_key(target))

        assert first > 0
        assert target.writes == []

    def test_one_changed_chunk_costs_the_same_however_tall_the_dataset(self):
        counts = []
        for rows in (64, 256, 1024):
            ds = dataset(rows)
            source = {}
            ds.to_zarr(store=source, mode="w", zarr_format=ZARR_FORMAT)
            target = CountingStore()
            reset_disk_persist_cache()
            _copy_changed_keys(source, target, _store_cache_key(target))

            # Re-encode one newly acquired row.
            ds["signal"].values[rows // 2, :] = np.random.rand(64)
            ds.to_zarr(store=source, mode="w", zarr_format=ZARR_FORMAT)
            target.writes.clear()
            _copy_changed_keys(source, target, _store_cache_key(target))
            counts.append(len(target.writes))

        assert len(set(counts)) == 1, (
            f"disk checkpoint scales with dataset height: {counts} keys written "
            f"for 64, 256 and 1024 rows"
        )


class TestRegionSafety:
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
        ds = xr.Dataset(
            {"signal": (("x", "y"), np.zeros((4, 2)))},
            coords={"x": np.array([0.0, 1.0, 1.0, 0.0]), "y": np.arange(2.0)},
        )

        assert _memory_region(ds, {"x": 1.0}, 0) is None

    def test_a_known_position_yields_the_rows_since_the_last_flush(self):
        ds = dataset(32)

        assert _memory_region(ds, {"x": 20.0}, 17) == (17, 21, "x")

    def test_a_position_behind_the_flush_point_is_still_covered(self):
        ds = dataset(32)

        assert _memory_region(ds, {"x": 5.0}, 20) == (5, 6, "x")
