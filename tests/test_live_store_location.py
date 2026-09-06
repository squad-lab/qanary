"""
Tests for where the live Zarr store lives and how it is cleaned up.

The store is scratch space for a running measurement: Qimchi reads it only as
a fallback while the run is in flight. Keeping it out of the data directory is
what stops it appearing in Qimchi's Explorer beside finished measurements, so
these pin the location, the happy-path cleanup, what happens to the data when
the export fails, and that pruning never touches a live run.

"""

from __future__ import annotations

import os
from itertools import count
from pathlib import Path
from unittest.mock import patch

import pytest
import xarray as xr
import zarr

from qcutils import measure
from qcutils.measure import (
    _live_store_root,
    _prune_orphaned_live_stores,
    _target_marker,
)
from qcutils.sweep import Sweep


@pytest.fixture
def seeded(measurement, gates, signal):
    """
    A measurement with a dataset and a live store, as it would be part-way
    through a run.

    Returns:
        xr.Dataset: The in-memory dataset.

    """
    dataset = measurement._make_dataset([Sweep(gates.x, 0.0, 1.0, num=3)], [signal])
    dataset["signal"].values[:] = [1.0, 2.0, 3.0]
    measurement.arr = dataset
    dataset.to_zarr(measurement.live_data, mode="w")
    return dataset


def _store(path: Path, points: int = 3) -> None:
    xr.Dataset(
        {"signal": ("x", [float(i) for i in range(points)])},
        coords={"x": list(range(points))},
    ).to_zarr(store=zarr.DirectoryStore(str(path)), mode="w")


class TestLocation:
    def test_the_root_is_under_the_qcutils_home(self, tmp_path):
        assert _live_store_root() == tmp_path / "qcutils-home" / "live"
        assert _live_store_root().is_dir()

    def test_a_measurement_puts_its_store_there_not_in_the_data_directory(
        self, measurement
    ):
        live = Path(measurement.live_data)

        assert live.parent == _live_store_root()
        assert live.name == f"{measurement.id}.zarr"
        assert Path(measurement.datalogging) not in live.parents

    def test_the_final_netcdf_still_lands_in_the_data_directory(self, measurement):
        assert Path(measurement.data).parent == Path(measurement.datalogging)


class TestPruning:
    def test_it_removes_a_store_whose_measurement_is_not_live(self, caplog):
        orphan = _live_store_root() / "1-abandoned.zarr"
        _store(orphan)
        os.utime(orphan, (0, 0))

        with patch.object(measure, "get_live_measurements", return_value=[]):
            _prune_orphaned_live_stores()

        assert not orphan.exists()
        assert "unfinished measurement" in caplog.text

    def test_it_leaves_a_live_measurement_alone(self):
        """A multi-day sweep outlives the age floor; the registry is what saves it."""
        running = _live_store_root() / "1-running.zarr"
        _store(running)
        os.utime(running, (0, 0))
        record = type("Row", (), {"measurement_id": "1-running"})()

        with patch.object(measure, "get_live_measurements", return_value=[record]):
            _prune_orphaned_live_stores()

        assert running.exists()

    def test_it_leaves_a_recently_created_store_alone(self):
        """Covers the window between creating a store and writing its row."""
        fresh = _live_store_root() / "1-just-started.zarr"
        _store(fresh)

        with patch.object(measure, "get_live_measurements", return_value=[]):
            _prune_orphaned_live_stores()

        assert fresh.exists()

    def test_an_unreadable_registry_prunes_nothing(self, caplog):
        orphan = _live_store_root() / "1-abandoned.zarr"
        _store(orphan)
        os.utime(orphan, (0, 0))

        with patch.object(
            measure, "get_live_measurements", side_effect=RuntimeError("locked")
        ):
            _prune_orphaned_live_stores()

        assert orphan.exists()
        assert "registry unreadable" in caplog.text


class TestFailedExportRescue:
    def test_the_store_is_converted_into_the_data_directory(self, measurement, caplog):
        """
        A failed export must not lose the run. Only the first write fails
        here, so the recovery conversion gets to succeed -- the case where a
        transient problem cost the export but not the machine.

        """
        _store(Path(measurement.live_data), points=4)
        real_to_netcdf = xr.Dataset.to_netcdf
        writes = count()

        def fail_the_first_write(self, *args, **kwargs):
            if next(writes) == 0:
                raise RuntimeError("transient lock")
            return real_to_netcdf(self, *args, **kwargs)

        with patch.object(xr.Dataset, "to_netcdf", fail_the_first_write):
            measurement._finalize_disk_artifacts(
                dataset=xr.Dataset({"a": ("i", [1.0])})
            )

        assert Path(measurement.data).exists()
        assert xr.load_dataset(measurement.data)["signal"].size == 4
        assert not Path(measurement.live_data).exists()
        assert "very likely incomplete" in caplog.text

    def test_an_unconvertible_store_is_still_preserved_beside_the_data(
        self, measurement, caplog
    ):
        """
        Conversion writes the same file the export just failed to write, so it
        can fail identically. The store has to survive that.

        """
        _store(Path(measurement.live_data))

        with (
            patch.object(
                xr.Dataset, "to_netcdf", side_effect=RuntimeError("disk full")
            ),
            patch.object(
                measure, "_convert_live_store", side_effect=RuntimeError("disk full")
            ),
        ):
            measurement._finalize_disk_artifacts(
                dataset=xr.Dataset({"a": ("i", [1.0])})
            )

        rescued = Path(measurement.datalogging) / f"{measurement.id}.zarr"
        assert rescued.exists(), "the only copy of the run must survive"
        assert not Path(measurement.live_data).exists()
        assert "qcutils.dataset.convert" in caplog.text

    def test_a_failed_export_with_no_store_reports_the_loss(self, measurement, caplog):
        with patch.object(
            xr.Dataset, "to_netcdf", side_effect=RuntimeError("disk full")
        ):
            result = measurement._finalize_disk_artifacts(
                dataset=xr.Dataset({"a": ("i", [1.0])})
            )

        assert result is None
        assert "Its data is lost" in caplog.text


class TestHappyPath:
    def test_a_successful_export_removes_the_store(self, measurement, seeded):
        _store(Path(measurement.live_data))

        measurement._finalize_disk_artifacts(dataset=seeded)

        assert Path(measurement.data).exists()
        assert not Path(measurement.live_data).exists()

    def test_nothing_is_left_in_the_data_directory_but_the_netcdf(
        self, measurement, seeded
    ):
        _store(Path(measurement.live_data))

        measurement._finalize_disk_artifacts(dataset=seeded)

        leftovers = {p.name for p in Path(measurement.datalogging).iterdir()}
        assert leftovers == {f"{measurement.id}.nc"}


class TestCrashRecovery:
    """
    A run killed outright never finalises, so its live store is the only copy
    of the data. The marker beside it says where that data belongs.

    """

    def _abandoned(self, tmp_path, points: int = 5) -> tuple[Path, Path]:
        """Leave a store and its marker behind, aged past the prune floor."""
        store = _live_store_root() / "1-crashed.zarr"
        _store(store, points=points)
        target = tmp_path / "data" / "W1" / "1-crashed.nc"
        _target_marker(store).write_text(str(target), encoding="utf-8")
        os.utime(store, (0, 0))
        return store, target

    def test_a_measurement_records_where_its_data_belongs(self, measurement):
        marker = _target_marker(Path(measurement.live_data))

        assert marker.read_text(encoding="utf-8") == str(
            Path(measurement.data).resolve()
        )

    def test_an_abandoned_store_is_recovered_into_its_data_directory(
        self, tmp_path, caplog
    ):
        store, target = self._abandoned(tmp_path)

        with patch.object(measure, "get_live_measurements", return_value=[]):
            _prune_orphaned_live_stores()

        assert target.exists(), "the only copy of the run must be recovered"
        assert xr.load_dataset(target)["signal"].size == 5
        assert not store.exists()
        assert not _target_marker(store).exists()
        assert "very likely incomplete" in caplog.text

    def test_a_store_with_no_marker_is_removed_rather_than_kept_forever(self, caplog):
        store = _live_store_root() / "1-unmarked.zarr"
        _store(store)
        os.utime(store, (0, 0))

        with patch.object(measure, "get_live_measurements", return_value=[]):
            _prune_orphaned_live_stores()

        assert not store.exists()
        assert "could not be recovered" in caplog.text

    def test_recovery_never_overwrites_an_existing_export(self, tmp_path, caplog):
        """A finished .nc is authoritative; a stale store must not clobber it."""
        store, target = self._abandoned(tmp_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        xr.Dataset({"signal": ("x", [9.0])}, coords={"x": [0]}).to_netcdf(target)

        with patch.object(measure, "get_live_measurements", return_value=[]):
            _prune_orphaned_live_stores()

        assert xr.load_dataset(target)["signal"].values.tolist() == [9.0]
        assert not store.exists()

    def test_a_live_measurement_is_never_recovered_from_under_itself(self, tmp_path):
        store, target = self._abandoned(tmp_path)
        record = type("Row", (), {"measurement_id": "1-crashed"})()

        with patch.object(measure, "get_live_measurements", return_value=[record]):
            _prune_orphaned_live_stores()

        assert store.exists()
        assert not target.exists()

    def test_a_successful_run_leaves_no_marker_behind(self, measurement, seeded):
        marker = _target_marker(Path(measurement.live_data))
        assert marker.exists()

        measurement._finalize_disk_artifacts(dataset=seeded)

        assert not marker.exists()
        assert list(_live_store_root().iterdir()) == []
