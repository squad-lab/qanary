"""
Tests for ``qcutils.dataset``.

These are the functions a user reaches for after a measurement, often after
something went wrong with it, so the error paths carry as much weight as the
happy ones. ``convert`` in particular is a recovery tool -- it exists for the
case where a run crashed before the framework turned its live ``.zarr`` store
into a ``.nc`` file -- and it must not destroy the Zarr store it is recovering
from unless the netCDF file is actually on disk.

"""

import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from qcutils.dataset import (
    DatasetConvertError,
    DatasetLoadError,
    DatasetNotFoundError,
    QCUtilsDatasetError,
    ZarrStoreNotFoundError,
    convert,
    load,
)

# ``load`` lets xarray choose the backend, matching how ``to_netcdf`` writes.
# These tests therefore run against whichever netCDF backend is installed
# (h5netcdf in a stock environment).


@pytest.fixture
def measurement() -> xr.Dataset:
    """
    A dataset shaped like a small two-dimensional sweep.

    Returns:
        xr.Dataset: One dependent over two coordinates, with attributes.

    """
    return xr.Dataset(
        {"dmm_v1": (("dac_ch1", "dac_ch2"), np.arange(6.0).reshape(2, 3))},
        coords={"dac_ch1": [0.0, 1.0], "dac_ch2": [0.0, 0.5, 1.0]},
        attrs={"Measurement ID": "1-abc", "Sample Name": "S1"},
    )


class TestLoad:
    """``load`` picks the engine from the path and reports failures plainly."""

    def test_reads_back_a_netcdf_file(self, tmp_path, measurement):
        path = tmp_path / "1-abc.nc"
        measurement.to_netcdf(path)

        xr.testing.assert_identical(load(path), measurement)

    def test_reads_back_a_zarr_store(self, tmp_path, measurement):
        """A live store is a directory, and the suffix selects the engine."""
        path = tmp_path / "1-abc.zarr"
        measurement.to_zarr(path)

        xr.testing.assert_identical(load(path), measurement)

    def test_accepts_a_string_path(self, tmp_path, measurement):
        path = tmp_path / "1-abc.nc"
        measurement.to_netcdf(path)

        assert list(load(str(path)).data_vars) == ["dmm_v1"]

    def test_missing_path_names_the_path_it_looked_for(self, tmp_path):
        missing = tmp_path / "never-ran.nc"

        with pytest.raises(DatasetNotFoundError, match="never-ran.nc"):
            load(missing)

    def test_unreadable_file_is_reported_as_a_load_failure(self, tmp_path):
        """
        A truncated or half-written file is the common real case, and the
        raw engine traceback is not what a measurement script should surface.

        """
        broken = tmp_path / "truncated.nc"
        broken.write_bytes(b"not a netCDF file")

        with pytest.raises(DatasetLoadError, match="Failed to load dataset"):
            load(broken)

    def test_load_errors_chain_the_underlying_cause(self, tmp_path):
        broken = tmp_path / "truncated.nc"
        broken.write_bytes(b"not a netCDF file")

        with pytest.raises(DatasetLoadError) as raised:
            load(broken)

        assert raised.value.__cause__ is not None


class TestConvert:
    """Recovering a ``.nc`` file from a leftover live ``.zarr`` store."""

    @pytest.fixture
    def stranded(self, tmp_path, measurement) -> Path:
        """
        A measurement whose zarr store survived but whose netCDF never got
        written -- exactly the state a crashed run leaves behind.

        Returns:
            Path: The ``.nc`` path that does not exist yet.

        """
        nc_path = tmp_path / "3-stranded.nc"
        measurement.to_zarr(nc_path.with_suffix(".zarr"))
        return nc_path

    def test_writes_the_netcdf_and_removes_the_zarr_store(self, stranded, measurement):
        result = convert(stranded)

        assert result == stranded.resolve()
        assert stranded.exists()
        assert not stranded.with_suffix(".zarr").exists()
        xr.testing.assert_identical(xr.load_dataset(stranded), measurement)

    def test_leaves_no_temporary_file_behind(self, stranded):
        convert(stranded)

        assert list(stranded.parent.glob("*.tmp")) == []

    def test_missing_zarr_store_is_reported(self, tmp_path):
        with pytest.raises(ZarrStoreNotFoundError, match="No .zarr store"):
            convert(tmp_path / "4-nothing.nc")

    def test_refuses_to_overwrite_an_existing_netcdf(self, stranded, measurement):
        """
        The finished file is the one with the data; clobbering it with a
        partial zarr store would be the wrong way round.

        """
        measurement.to_netcdf(stranded)

        with pytest.raises(DatasetConvertError, match="already exists"):
            convert(stranded)

        assert stranded.with_suffix(".zarr").exists(), "the source must survive"

    def test_overwrites_when_asked(self, stranded, measurement):
        stranded.write_bytes(b"stale placeholder")

        convert(stranded, overwrite=True)

        xr.testing.assert_identical(xr.load_dataset(stranded), measurement)

    def test_unreadable_zarr_store_is_reported(self, tmp_path):
        """A directory named ``.zarr`` is not necessarily a zarr store."""
        nc_path = tmp_path / "5-corrupt.nc"
        (tmp_path / "5-corrupt.zarr").mkdir()

        with pytest.raises(DatasetConvertError, match="Failed to read zarr store"):
            convert(nc_path)

    def test_failed_write_cleans_up_and_keeps_the_source(
        self, tmp_path, stranded, monkeypatch
    ):
        """
        A full disk must not leave a ``.nc.tmp`` stub that a later run mistakes
        for output, and must not delete the zarr store the data still lives in.

        """

        def explode(*args, **kwargs):
            Path(args[0]).write_bytes(b"partial")
            raise OSError("no space left on device")

        monkeypatch.setattr(xr.Dataset, "to_netcdf", explode)

        with pytest.raises(DatasetConvertError, match="Failed to write netCDF"):
            convert(stranded)

        assert not stranded.exists()
        assert list(tmp_path.glob("*.tmp")) == []
        assert stranded.with_suffix(".zarr").exists()

    def test_undeletable_zarr_store_does_not_fail_the_conversion(
        self, stranded, monkeypatch, caplog
    ):
        """
        The netCDF file is the product; a zarr store left behind because
        Windows still holds a handle on it is a cleanup nuisance, not a
        failure.

        """
        monkeypatch.setattr(
            shutil, "rmtree", lambda *a, **k: (_ for _ in ()).throw(OSError("locked"))
        )

        result = convert(stranded)

        assert result.exists()
        assert stranded.with_suffix(".zarr").exists()
        assert "failed to remove zarr store" in caplog.text

    def test_accepts_a_string_path(self, stranded):
        assert convert(str(stranded)).exists()


def test_every_dataset_error_shares_one_base_class():
    """
    A script that only wants to know "the dataset step failed" should be able
    to catch one exception rather than five.

    """
    for error in (
        DatasetNotFoundError,
        ZarrStoreNotFoundError,
        DatasetLoadError,
        DatasetConvertError,
    ):
        assert issubclass(error, QCUtilsDatasetError)


def test_an_undeletable_temporary_file_does_not_mask_the_write_error(
    tmp_path, measurement, monkeypatch
):
    """
    The cleanup of a failed conversion is itself best-effort: what the caller
    needs to see is why the write failed, not why the stub could not be
    removed.

    """
    nc_path = tmp_path / "6-stubborn.nc"
    measurement.to_zarr(nc_path.with_suffix(".zarr"))

    def explode(self, path, *args, **kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("no space left on device")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", explode)
    monkeypatch.setattr(
        Path, "unlink", lambda *a, **k: (_ for _ in ()).throw(OSError("locked"))
    )

    with pytest.raises(DatasetConvertError, match="no space left on device"):
        convert(nc_path)
