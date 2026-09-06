from __future__ import annotations

import shutil
from pathlib import Path
from typing import Union

import xarray as xr

# Local imports
from qcutils.logger import get_logger

# Public API
__all__ = [
    "load",
    "convert",
    "QCUtilsDatasetError",
    "DatasetNotFoundError",
    "ZarrStoreNotFoundError",
    "DatasetLoadError",
    "DatasetConvertError",
]


logger = get_logger(__name__)


# Custom exceptions
class QCUtilsDatasetError(Exception):
    """Base class for all dataset-related errors in qcutils."""


class DatasetNotFoundError(QCUtilsDatasetError):
    """Raised when the requested dataset file or directory does not exist."""


class ZarrStoreNotFoundError(QCUtilsDatasetError):
    """Raised when the expected .zarr store directory cannot be found."""


class DatasetLoadError(QCUtilsDatasetError):
    """Raised when xarray cannot open the dataset."""


class DatasetConvertError(QCUtilsDatasetError):
    """Raised when conversion from .zarr to .nc fails."""


# Public helpers
def load(path: Union[str, Path]) -> xr.Dataset:
    """
    Load a qcutils measurement dataset.

    Wraps :func:`xarray.load_dataset` with engine selection appropriate for
    QCUtils-generated files (automatic selection for ``.nc`` and the Zarr
    engine for ``.zarr``) and clear error messages.

    Args:
        path (Union[str, Path]): Path to a ``.nc`` file or ``.zarr`` directory
            created by a QCUtils :class:`~qcutils.measure.Measurement` run.

    Returns:
        xr.Dataset: The fully-loaded (in-memory) xarray dataset.

    Raises:
        DatasetNotFoundError: If *path* does not exist on disk.
        DatasetLoadError: If xarray raises any error while opening the file.

    """
    path = Path(path)

    if not path.exists():
        raise DatasetNotFoundError(
            f"Dataset not found: '{path}'. "
            "Check that the path is correct and the measurement has finished."
        )

    engine = "zarr" if path.suffix == ".zarr" else None

    logger.info(f"Loading dataset from '{path}' (engine={engine or 'auto'})")

    try:
        ds = xr.load_dataset(path, engine=engine)
    except Exception as exc:
        raise DatasetLoadError(f"Failed to load dataset from '{path}': {exc}") from exc

    logger.info(f"Loaded dataset: {list(ds.data_vars)} | dims={dict(ds.dims)}")
    return ds


def convert(nc_path: Union[str, Path], *, overwrite: bool = False) -> Path:
    """
    Convert the paired ``.zarr`` store for a measurement to netCDF.

    A normal run exports its current in-memory dataset directly to netCDF and
    removes its temporary live checkpoint. If recovery instead leaves a
    ``.zarr`` directory beside the intended ``.nc`` file, this function can
    complete that conversion manually.

    Expected layout on disk::

        <stem>.nc      <- target output (may or may not exist yet)
        <stem>.zarr/   <- live zarr store (must exist)

    Args:
        nc_path (Union[str, Path]): Path to the (possibly missing) ``.nc`` file.
            The function derives the ``.zarr`` path by replacing the suffix.
        overwrite (bool): If *True*, overwrite an existing ``.nc`` file silently.
            Defaults to *False*.

    Returns:
        Path: Resolved absolute path to the written ``.nc`` file.

    Raises:
        ZarrStoreNotFoundError: If the ``.zarr`` directory does not exist next
            to *nc_path*.
        DatasetConvertError: If *nc_path* already exists and *overwrite* is
            *False*, or if conversion fails for another reason.

    """
    nc_path = Path(nc_path).resolve()
    zarr_path = nc_path.with_suffix(".zarr")

    # Validate zarr store
    if not zarr_path.exists():
        raise ZarrStoreNotFoundError(
            f"No .zarr store found at '{zarr_path}'. "
            "Either the measurement never started writing, or the zarr store "
            "was already converted and removed."
        )

    # Guard against accidental overwrites
    if nc_path.exists() and not overwrite:
        raise DatasetConvertError(
            f"Output file '{nc_path}' already exists. "
            "Pass overwrite=True to replace it, or remove the file manually."
        )

    # Load zarr → write netCDF
    logger.info(f"Converting '{zarr_path}' -> '{nc_path}'")

    try:
        ds = xr.load_dataset(zarr_path, engine="zarr")
    except Exception as exc:
        raise DatasetConvertError(
            f"Failed to read zarr store at '{zarr_path}': {exc}"
        ) from exc

    tmp_nc = nc_path.with_suffix(".nc.tmp")
    try:
        ds.to_netcdf(tmp_nc, mode="w")
        tmp_nc.replace(nc_path)
    except Exception as exc:
        # Clean up incomplete temporary file
        if tmp_nc.exists():
            try:
                tmp_nc.unlink()
            except OSError:
                pass
        raise DatasetConvertError(
            f"Failed to write netCDF file to '{nc_path}': {exc}"
        ) from exc

    logger.info(f"Successfully wrote netCDF file: '{nc_path}'")

    # Remove zarr store
    try:
        shutil.rmtree(zarr_path)
        logger.info(f"Removed zarr store: '{zarr_path}'")
    except Exception as exc:
        # Non-fatal: warn but don't fail the whole operation
        logger.warning(
            f"netCDF export succeeded but failed to remove zarr store "
            f"at '{zarr_path}': {exc}. You may delete it manually."
        )

    return nc_path
