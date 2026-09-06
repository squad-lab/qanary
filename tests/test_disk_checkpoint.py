"""
Tests for incremental disk checkpointing.

A checkpoint used to copy every key in the Zarr store, which is dominated by
per-file I/O rather than data volume while only the chunk just acquired actually
differs. These pin the behaviour that replaced it: the bytes on disk must stay
identical to a full copy, and a failed write must stay dirty so a retry picks it
up.

"""

import hashlib
import os
import shutil
import tempfile

import numpy as np
import pytest
import xarray as xr
import zarr

from qanary.sweep import (
    _copy_changed_keys,
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


def _store(tag: str) -> zarr.DirectoryStore:
    """Create an empty DirectoryStore in a throwaway directory."""
    return zarr.DirectoryStore(os.path.join(tempfile.mkdtemp(), f"{tag}.zarr"))


def _tree_hash(store) -> str:
    """Hash every key and value in a store, so two stores must match exactly."""
    digest = hashlib.sha256()
    for key in sorted(store.keys()):
        digest.update(key.encode())
        digest.update(store[key])
    return digest.hexdigest()


def test_incremental_copy_matches_a_full_copy(dataset):
    """Skipping unchanged keys must not change what ends up on disk."""
    memory = zarr.MemoryStore()
    dataset.to_zarr(store=memory, mode="w")
    reference, incremental = _store("reference"), _store("incremental")
    reset_disk_persist_cache(incremental)

    for step in range(3 * N):
        dataset["dmm_v1"].values[(step // N) % N, step % N] = float(step)
        dataset.to_zarr(store=memory, mode="w")
        zarr.copy_store(memory, reference, if_exists="replace")
        _copy_changed_keys(memory, incremental, _store_cache_key(incremental))

        assert _tree_hash(reference) == _tree_hash(incremental)

    restored = xr.open_zarr(incremental.path, consolidated=False).load()
    xr.testing.assert_identical(
        restored, xr.open_zarr(reference.path, consolidated=False).load()
    )


def test_only_changed_keys_are_rewritten(dataset):
    """The point of the exercise: one acquired chunk, one file written."""
    memory = zarr.MemoryStore()
    dataset.to_zarr(store=memory, mode="w")
    target = _store("target")
    reset_disk_persist_cache(target)

    first = _copy_changed_keys(memory, target, _store_cache_key(target))
    assert first == len(memory), "the first checkpoint must write the whole store"

    unchanged = _copy_changed_keys(memory, target, _store_cache_key(target))
    assert unchanged == 0, "an unchanged store must write nothing"

    dataset["dmm_v1"].values[0, 0] = 1.0
    dataset.to_zarr(store=memory, mode="w")
    changed = _copy_changed_keys(memory, target, _store_cache_key(target))
    assert changed < len(memory)


def test_store_removed_underneath_is_rebuilt(dataset):
    """A vanished store must be rewritten in full, not skipped as up to date."""
    memory = zarr.MemoryStore()
    dataset.to_zarr(store=memory, mode="w")
    target = _store("vanishing")
    reset_disk_persist_cache(target)
    _copy_changed_keys(memory, target, _store_cache_key(target))

    shutil.rmtree(target.path)
    written = _copy_changed_keys(memory, target, _store_cache_key(target))

    assert written == len(memory)


def test_failed_write_is_retried(dataset):
    """A locked key must stay dirty so the next attempt writes it."""

    class LockedOnce:
        """Store proxy that raises PermissionError once for one key."""

        def __init__(self, real, fail_on):
            self.real, self.fail_on, self.armed = real, fail_on, True

        def __contains__(self, key):
            return key in self.real

        def keys(self):
            return self.real.keys()

        def __getitem__(self, key):
            return self.real[key]

        def __setitem__(self, key, value):
            if self.armed and key == self.fail_on:
                self.armed = False
                raise PermissionError("store is locked")
            self.real[key] = value

    memory = zarr.MemoryStore()
    dataset.to_zarr(store=memory, mode="w")
    target = _store("locked")
    reset_disk_persist_cache(target)

    locked_key = "dmm_v1/0.0"
    with pytest.raises(PermissionError):
        _copy_changed_keys(
            memory, LockedOnce(target, locked_key), _store_cache_key(target)
        )

    # As _persist_disk's retry loop does.
    _copy_changed_keys(memory, target, _store_cache_key(target))

    reference = _store("reference")
    zarr.copy_store(memory, reference, if_exists="replace")
    assert _tree_hash(reference) == _tree_hash(target)


def test_reset_clears_only_the_named_store(dataset):
    """Resetting one measurement must not force a rewrite of another."""
    memory = zarr.MemoryStore()
    dataset.to_zarr(store=memory, mode="w")
    kept, cleared = _store("kept"), _store("cleared")
    for store in (kept, cleared):
        reset_disk_persist_cache(store)
        _copy_changed_keys(memory, store, _store_cache_key(store))

    reset_disk_persist_cache(cleared)

    assert _copy_changed_keys(memory, kept, _store_cache_key(kept)) == 0
    assert _copy_changed_keys(memory, cleared, _store_cache_key(cleared)) == len(memory)
