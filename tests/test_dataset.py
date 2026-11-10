"""Tests for loading, converting, and recovering datasets."""

import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from qanary.dataset import (
    DatasetConvertError,
    DatasetLoadError,
    DatasetNotFoundError,
    QanaryDatasetError,
    ZarrStoreNotFoundError,
    convert,
    load,
)

# ``load`` lets xarray choose the backend, matching how ``to_netcdf`` writes.
# These tests therefore run against whichever netCDF backend is installed
# (h5netcdf in a stock environment).


@pytest.fixture
def measurement() -> xr.Dataset:
    """Return a small two-dimensional measurement dataset."""
    return xr.Dataset(
        {"dmm_v1": (("dac_ch1", "dac_ch2"), np.arange(6.0).reshape(2, 3))},
        coords={"dac_ch1": [0.0, 1.0], "dac_ch2": [0.0, 0.5, 1.0]},
        attrs={"Measurement ID": "1-abc", "Sample Name": "S1"},
    )


class TestLoad:
    def test_reads_back_a_netcdf_file(self, tmp_path, measurement):
        path = tmp_path / "1-abc.nc"
        measurement.to_netcdf(path)

        xr.testing.assert_identical(load(path), measurement)

    def test_reads_back_a_zarr_store(self, tmp_path, measurement):
        path = tmp_path / "1-abc.zarr"
        measurement.to_zarr(path, zarr_format=2)

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
    @pytest.fixture
    def stranded(self, tmp_path, measurement) -> Path:
        nc_path = tmp_path / "3-stranded.nc"
        measurement.to_zarr(nc_path.with_suffix(".zarr"), zarr_format=2)
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
        measurement.to_netcdf(stranded)

        with pytest.raises(DatasetConvertError, match="already exists"):
            convert(stranded)

        assert stranded.with_suffix(".zarr").exists(), "the source must survive"

    def test_overwrites_when_asked(self, stranded, measurement):
        stranded.write_bytes(b"stale placeholder")

        convert(stranded, overwrite=True)

        xr.testing.assert_identical(xr.load_dataset(stranded), measurement)

    def test_unreadable_zarr_store_is_reported(self, tmp_path):
        nc_path = tmp_path / "5-corrupt.nc"
        (tmp_path / "5-corrupt.zarr").mkdir()

        with pytest.raises(DatasetConvertError, match="Failed to read zarr store"):
            convert(nc_path)

    def test_failed_write_cleans_up_and_keeps_the_source(
        self, tmp_path, stranded, monkeypatch
    ):

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
    for error in (
        DatasetNotFoundError,
        ZarrStoreNotFoundError,
        DatasetLoadError,
        DatasetConvertError,
    ):
        assert issubclass(error, QanaryDatasetError)


def test_an_undeletable_temporary_file_does_not_mask_the_write_error(
    tmp_path, measurement, monkeypatch
):
    nc_path = tmp_path / "6-stubborn.nc"
    measurement.to_zarr(nc_path.with_suffix(".zarr"), zarr_format=2)

    def explode(self, path, *args, **kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("no space left on device")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", explode)
    monkeypatch.setattr(
        Path, "unlink", lambda *a, **k: (_ for _ in ()).throw(OSError("locked"))
    )

    with pytest.raises(DatasetConvertError, match="no space left on device"):
        convert(nc_path)
