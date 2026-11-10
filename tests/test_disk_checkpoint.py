"""Tests for incremental disk checkpoints."""

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

from qanary.sweep import (
    ZARR_FORMAT,
    _copy_changed_keys,
    _copy_store,
    _store_cache_key,
    reset_disk_persist_cache,
)

N = 8


@pytest.fixture
def dataset() -> xr.Dataset:
    """A small pre-allocated grid, shaped like a live sweep."""
    return xr.Dataset(
        {"dmm_v1": (("dac_ch1", "dac_ch2"), np.full((N, N), np.nan))},
        coords={
            "dac_ch1": np.linspace(0, 1, N),
            "dac_ch2": np.linspace(0, 1, N),
        },
        attrs={"Sweeps": "{}"},
    )


def _store(tag: str) -> zarr.storage.LocalStore:
    """Create an empty LocalStore in a throwaway directory."""
    return zarr.storage.LocalStore(os.path.join(tempfile.mkdtemp(), f"{tag}.zarr"))


def _seeded(dataset: xr.Dataset) -> dict:
    """Write a dataset to an in-memory store, as ``Measurement.run`` does."""
    memory = {}
    dataset.to_zarr(store=memory, mode="w", zarr_format=ZARR_FORMAT)
    return memory


def _tree_hash(store) -> str:
    """Hash every file in a store, so two stores must match exactly."""
    root = Path(store.root)
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_incremental_copy_matches_a_full_copy(dataset):
    memory = _seeded(dataset)
    reference, incremental = _store("reference"), _store("incremental")
    reset_disk_persist_cache(incremental)

    for step in range(3 * N):
        dataset["dmm_v1"].values[(step // N) % N, step % N] = float(step)
        dataset.to_zarr(store=memory, mode="w", zarr_format=ZARR_FORMAT)
        _copy_store(memory, reference)
        _copy_changed_keys(memory, incremental, _store_cache_key(incremental))

        assert _tree_hash(reference) == _tree_hash(incremental)

    restored = xr.open_zarr(incremental.root, consolidated=False).load()
    xr.testing.assert_identical(
        restored, xr.open_zarr(reference.root, consolidated=False).load()
    )


def test_only_changed_keys_are_rewritten(dataset):
    memory = _seeded(dataset)
    target = _store("target")
    reset_disk_persist_cache(target)

    first = _copy_changed_keys(memory, target, _store_cache_key(target))
    assert first == len(memory), "the first checkpoint must write the whole store"

    unchanged = _copy_changed_keys(memory, target, _store_cache_key(target))
    assert unchanged == 0, "an unchanged store must write nothing"

    dataset["dmm_v1"].values[0, 0] = 1.0
    dataset.to_zarr(store=memory, mode="w", zarr_format=ZARR_FORMAT)
    changed = _copy_changed_keys(memory, target, _store_cache_key(target))
    assert changed < len(memory)


def test_store_removed_underneath_is_rebuilt(dataset):
    memory = _seeded(dataset)
    target = _store("vanishing")
    reset_disk_persist_cache(target)
    _copy_changed_keys(memory, target, _store_cache_key(target))

    shutil.rmtree(target.root)
    written = _copy_changed_keys(memory, target, _store_cache_key(target))

    assert written == len(memory)


def test_failed_write_is_retried(dataset):

    class LockedOnce(zarr.storage.LocalStore):
        def __init__(self, root, fail_on):
            super().__init__(root)
            self.fail_on, self.armed = fail_on, True

        async def set(self, key, value, *args, **kwargs):
            if self.armed and key == self.fail_on:
                self.armed = False
                raise PermissionError("store is locked")
            await super().set(key, value, *args, **kwargs)

    # Zarr does not store a chunk that holds only the fill value.
    dataset["dmm_v1"].values[0, 0] = 0.0
    memory = _seeded(dataset)
    target = _store("locked")
    reset_disk_persist_cache(target)

    locked_key = "dmm_v1/0.0"
    assert locked_key in memory
    with pytest.raises(PermissionError):
        _copy_changed_keys(
            memory, LockedOnce(target.root, locked_key), _store_cache_key(target)
        )

    # Match `_persist_disk` by retrying after the lock clears.
    _copy_changed_keys(memory, target, _store_cache_key(target))

    reference = _store("reference")
    _copy_store(memory, reference)
    assert _tree_hash(reference) == _tree_hash(target)


def test_reset_clears_only_the_named_store(dataset):
    memory = _seeded(dataset)
    kept, cleared = _store("kept"), _store("cleared")
    for store in (kept, cleared):
        reset_disk_persist_cache(store)
        _copy_changed_keys(memory, store, _store_cache_key(store))

    reset_disk_persist_cache(cleared)

    assert _copy_changed_keys(memory, kept, _store_cache_key(kept)) == 0
    assert _copy_changed_keys(memory, cleared, _store_cache_key(cleared)) == len(memory)
