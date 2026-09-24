import datetime
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import uuid
from pathlib import Path
from time import time
from typing import Callable, Optional, Sequence, Union

import numpy as np
import xarray as xr
import zarr
from checksumdir import dirhash
from qcodes.parameters import Parameter
from qimchi_connect import (
    QanarySnapshotProvider,
    close_live_measurement,
    get_live_measurements,
    register_live_measurement,
    update_live_disk_path,
)
from tqdm import tqdm

# Local imports
from qanary.buffered.sweep import (
    _abort_instruments,
    _buffered_sweep_progress_info,
    _fetch_dependents_tree,
)
from qanary.dataset import convert as _convert_live_store
from qanary.logger import get_logger
from qanary.parameters import MultiChannelParameter, ParameterMixin
from qanary.sweep import (
    CircularSweep,
    Sweep,
    _stepper,
    _sweep_parameters,
    reset_disk_persist_cache,
    sweeper,
)

__all__ = [
    "Station",
    "Measurement",
    "run",
]


logger = get_logger(__name__)


def _live_store_root() -> Path:
    """
    Return the directory holding live Zarr stores, creating it if needed.

    Each store is a temporary disk checkpoint for a running measurement and a
    fallback for live consumers. It is removed after a successful netCDF
    export.

    ``QANARY_HOME`` overrides the default ``~/.qanary`` application directory.

    Returns:
        Path: Directory for live Zarr stores.

    """
    home = os.environ.get("QANARY_HOME")
    directory = (Path(home).expanduser() if home else Path.home() / ".qanary") / "live"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _target_marker(store_path: Path) -> Path:
    """
    Return the sidecar file that records a live store's final netCDF path.

    Args:
        store_path (Path): The live Zarr store.

    Returns:
        Path: Marker path ending in ``.zarr.target``.

    """
    return store_path.with_suffix(".zarr.target")


def _recover_orphaned_store(store_path: Path) -> bool:
    """
    Convert an abandoned live store into the netCDF file it was headed for.

    Args:
        store_path (Path): Live Zarr store of a measurement that never
            finalised.

    Returns:
        bool: Whether conversion produced the intended netCDF file. A false
            result can mean that the target already exists, no target was
            recorded, or recovery failed.

    """
    marker = _target_marker(store_path)
    try:
        target = Path(marker.read_text(encoding="utf-8").strip())
    except Exception as exc:
        logger.warning(f"No recovery target recorded for {store_path}: {exc}")
        return False

    if target.exists():
        logger.info(f"{target} already exists; leaving it alone.")
        return False

    staged = target.with_suffix(".zarr")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(store_path), str(staged))
        _convert_live_store(target, overwrite=False)
    except Exception as exc:
        logger.error(
            f"Could not recover the interrupted measurement at {store_path} "
            f"into {target}: {exc}. Its data is still in the zarr store."
        )
        return False

    logger.warning(
        f"Recovered an interrupted measurement into {target}. It never "
        "finished, so it is very likely incomplete."
    )
    return True


def _prune_orphaned_live_stores(minimum_age_seconds: float = 3600.0) -> None:
    """
    Recover live Zarr stores left behind by measurements that are no longer live.

    An old store whose measurement is absent from the live registry is treated
    as abandoned, usually because its process ended before finalisation. The
    store may contain the only copy of the acquired data, so Qanary first tries
    to recover it to its intended netCDF file. If conversion fails after the
    store has been moved beside that target, the staged Zarr data is preserved
    for manual recovery. An unrecoverable scratch store is otherwise removed.

    New stores are left alone because a producer may create its store before
    publishing its discovery record. This grace period prevents a concurrent
    measurement from being mistaken for an abandoned one.

    Args:
        minimum_age_seconds (float): Leave stores younger than this alone.

    """
    try:
        live_ids = {record.measurement_id for record in get_live_measurements()}
    except Exception as exc:
        logger.warning(f"Skipped pruning live stores; registry unreadable: {exc}")
        return

    cutoff = time() - minimum_age_seconds
    for store_path in _live_store_root().glob("*.zarr"):
        if store_path.stem in live_ids:
            continue
        try:
            if store_path.stat().st_mtime > cutoff:
                continue
            if _recover_orphaned_store(store_path):
                _target_marker(store_path).unlink(missing_ok=True)
                continue
            if store_path.exists():
                logger.warning(
                    f"Removing the live store of an unfinished measurement that "
                    f"could not be recovered: {store_path}."
                )
                shutil.rmtree(store_path)
            _target_marker(store_path).unlink(missing_ok=True)
        except Exception as exc:
            logger.warning(f"Failed pruning orphaned live store {store_path}: {exc}")


def _register_memory_store(
    measurement_id: str, store: zarr.MemoryStore, disk_path: Optional[str] = None
) -> int:
    """
    Publish a memory store as a live measurement.

    Args:
        measurement_id (str): Stable identifier for the active measurement.
        store (zarr.MemoryStore): In-memory Zarr store containing live data.
        disk_path (Optional[str]): Persisted fallback location advertised to
            consumers.

    Returns:
        int: Port serving the live measurement, or zero if publication failed.

    """
    _prune_orphaned_live_stores()

    # One call sweeps stale rows, starts the server if it is not already up,
    # publishes the snapshot callback, and writes the discovery row Qimchi
    # reads. It also tracks the publication, so there is nothing to keep here.
    try:
        registration = register_live_measurement(
            measurement_id,
            QanarySnapshotProvider(store),
            disk_path=disk_path,
            port=_find_available_port(8765),
            retention_days=7,
        )
    except Exception as exc:
        logger.error(f"Failed publishing live measurement {measurement_id}: {exc}")
        return 0

    logger.info(f"Live measurement {measurement_id} published at {registration.ws_url}")
    return registration.ws_port


def _find_available_port(start_port: int, max_attempts: int = 100) -> int:
    """
    Find an available local TCP port in a bounded range.

    Args:
        start_port (int): First port to probe.
        max_attempts (int): Number of consecutive ports to probe.

    Returns:
        int: First available port, or *start_port* if every probe fails.

    """
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                return port
            except OSError:
                continue
    # Fallback to start_port if none found
    return start_port


def _unregister_memory_store(measurement_id: str) -> None:
    """
    Stop publishing a live measurement through Qimchi Connect.

    Closing the publication also removes its snapshot provider and marks its
    discovery record as ended. Failures are logged so measurement finalisation
    is not masked by cleanup errors.

    Args:
        measurement_id (str): Identifier of the publication to close.

    """
    try:
        # Stops advertising the measurement, drops its cached snapshot, and
        # marks the discovery record ended.
        if close_live_measurement(measurement_id):
            logger.info(f"Stopped publishing live measurement {measurement_id}")
    except Exception as e:
        logger.warning(f"Failed closing live measurement {measurement_id}: {e}")


bar = None


def _sanitize_for_json(obj):
    """
    Recursively convert snapshot values to JSON-serializable objects.

    Instrument snapshots may contain driver-specific context objects in
    parameter caches (for example ``QDac2Trigger_Context`` after a sweep).
    Preserve ordinary JSON values and stringify unsupported objects rather
    than failing the whole measurement metadata export.

    Args:
        obj: Snapshot value to sanitize.

    Returns:
        A JSON-compatible value.

    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(v) for v in obj]

    try:
        json.dumps(obj)
    except (TypeError, ValueError):
        return str(obj)

    return obj


class Station:
    """
    Collection of instruments and named parameters recorded by a measurement.

    Args:
        name (str): Human-readable station name.

    """

    def __init__(self, name: str):
        self.name = name
        self.instruments = []
        self.parameters = []

    def add_parameter(
        self,
        name: str,
        label: str,
        param: Parameter | Sequence[Parameter],
        param_type: str = "gate",
        override: bool = False,
    ):
        """
        Add a named parameter to the station.

        A sequence is represented by a
        :class:`~qanary.parameters.MultiChannelParameter`; a single QCoDeS
        parameter is aliased with :class:`~qanary.parameters.ParameterMixin`.

        Args:
            name (str): Name used in measurement metadata and datasets.
            label (str): Human-readable label.
            param (Parameter | Sequence[Parameter]): Parameter or channels to
                register.
            param_type (str): Parameter category stored in snapshots. Defaults
                to ``"gate"``.
            override (bool): Replace an existing parameter with the same

        Returns:
            ParameterMixin | MultiChannelParameter: Registered station
                parameter.

        Raises:
            ValueError: If another station parameter already uses *name* and override is False.

        """
        if isinstance(param, Sequence):
            pm = MultiChannelParameter(param, name, label, param_type)
        else:
            pm = ParameterMixin(param, name, label, param_type)

        # Compare by name, not by wrapper identity.
        for i, existing in enumerate(self.parameters):
            if existing.name == pm.name:
                if not override:
                    raise ValueError(
                        f"Parameter {pm.name} already exists in station {self.name}"
                    )

                self.parameters[i] = pm
                return pm

        self.parameters.append(pm)
        return pm

    def remove_parameter(self, pm: ParameterMixin):
        """
        Remove a previously registered parameter.

        Args:
            pm (ParameterMixin): Parameter returned by
                :meth:`add_parameter`.

        Raises:
            ValueError: If *pm* is not registered with this station.

        """
        if pm in self.parameters:
            self.parameters.remove(pm)
        else:
            raise ValueError(f"Parameter {pm.name} not found in station {self.name}")


class Measurement:
    """
    Configure, acquire, publish, and persist one Qanary measurement.

    """

    def __init__(
        self,
        wafer_id: str,
        device_type: str,
        sample_name: str,
        experiment_name: str,
        station: Station,
        data_location: str,
        metadata: dict,
        fridge_name: str = "",
        save_interval: float = 0.1,
        nc_snapshot_during_run: bool = False,  # @Spandan - edit as needed
        git_repo: str = "~/.measurement-hashes",
    ):
        """
        Create a measurement and reserve its output paths.

        Args:
            wafer_id (str): Wafer identifier used in the output path.
            device_type (str): Device type used in the output path.
            sample_name (str): Sample name used in the output path.
            experiment_name (str): Experiment name used in the output path.
            station (Station): Instruments and parameters to snapshot.
            data_location (str): Root directory for completed measurements.
            metadata (dict): User metadata stored on the resulting dataset.
            fridge_name (str): Optional suffix for the measurement-hash
                repository. Defaults to an empty string.
            save_interval (float): Minimum time in seconds between live and disk
                checkpoints. Defaults to 0.1.
            nc_snapshot_during_run (bool): Also write best-effort netCDF
                snapshots during acquisition. Defaults to False.
            git_repo (str): Repository used to record measurement hashes.
                Defaults to ``"~/.measurement-hashes"``.

        Notes:
            Completed data is written below
            ``data_location/wafer_id/device_type/sample_name/experiment_name``.
            Live recovery checkpoints are temporary and stored under the
            Qanary application directory (``QANARY_HOME``, ``~/.qanary`` by
            default).

        """
        self.wafer_id = wafer_id
        self.device_type = device_type
        self.sample_name = sample_name
        self.experiment = experiment_name
        self.extra_metadata = metadata

        self.id = f"1-{uuid.uuid4()}"
        self.datalogging = (
            f"{data_location}/{wafer_id}/{device_type}/{sample_name}/{experiment_name}"
        )
        os.makedirs(self.datalogging, exist_ok=True)

        self.save_interval = save_interval
        self.nc_snapshot_during_run = nc_snapshot_during_run
        if not fridge_name:
            self.git_repo = os.path.expanduser(git_repo)
        else:
            self.git_repo = os.path.expanduser(f"{git_repo}-{fridge_name}")

        data_files = os.listdir(self.datalogging)
        # Keeping .zarr for backward compatibility, but we will migrate to .nc in the future
        data_files = [
            int(file.split("-")[0])
            for file in data_files
            if file.endswith(".zarr") or file.endswith(".nc")
        ]

        if len(data_files) != 0:
            self.id = f"{sorted(data_files)[-1] + 1}-{uuid.uuid4()}"

        self.data = f"{self.datalogging}/{self.id}.nc"
        self.live_data = str(_live_store_root() / f"{self.id}.zarr")
        try:
            _target_marker(Path(self.live_data)).write_text(
                str(Path(self.data).resolve()), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning(f"Could not record the recovery target: {exc}")
        self.arr = None
        self.memory_store = None
        self.disk_store = None
        self.station = station
        self.fridge_name = fridge_name
        logger.info(f"Measurement Location: {self.data}")

    def get_installed_packages(self):
        """
        Return installed Python distributions as requirements text.

        Returns:
            str: ``name==version`` lines suitable for dataset metadata, or an
                empty string when neither package-listing command is available.

        """
        # uv-created environments do not necessarily contain the pip module.
        # Prefer uv, then quietly fall back to pip for non-uv environments.
        commands = (
            ["uv", "pip", "freeze"],
            [sys.executable, "-m", "pip", "freeze"],
        )
        for command in commands:
            try:
                return subprocess.check_output(
                    command,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue

        return ""

    def _make_dataarray(self, sweeps, dependent):
        """
        Preallocate a NaN-filled data array for one dependent parameter.

        Args:
            sweeps (Sequence[Sweep]): Sweeps defining the array dimensions and
                coordinates.
            dependent (Parameter): Measured QCoDeS parameter represented by the
                array.

        Returns:
            xr.DataArray: Data array with parameter and coordinate metadata.

        """
        data_array = xr.DataArray(
            data=np.empty(
                [len(sweep.values) for sweep in sweeps for param in sweep.parameter]
            ),
            coords=[sweep.values for sweep in sweeps for param in sweep.parameter],
            dims=[param.name for sweep in sweeps for param in sweep.parameter],
            attrs={
                "unit": dependent.unit,
                "label": dependent.label,
                "instrument": dependent.instrument.name,
                # snapshots included in the gloabl metadata
                # "instrument_snapshot": str(dependent.instrument.snapshot()),
            },
        )
        for sweep in sweeps:
            for param in sweep.parameter:
                data_array.coords[param.name].attrs["instrument"] = (
                    param.instrument.name
                )
                data_array.coords[param.name].attrs["unit"] = param.unit
                data_array.coords[param.name].attrs["label"] = param.label

        data_array.data[:] = np.nan
        return data_array

    def _make_dataset(self, sweeps: Sequence[Union[Sweep, dict]], dependents: list):
        """
        Preallocate the measurement dataset and its acquisition metadata.

        Args:
            sweeps (Sequence[Sweep | dict]): Slow sweeps, optionally followed by
                a buffered-sweep tree.
            dependents (list): Unbuffered QCoDeS parameters to include as data
                variables.

        Returns:
            xr.Dataset: Dataset containing NaN-filled dependent arrays and all
                sweep coordinates.

        """
        if isinstance(sweeps[-1], dict):
            buffered_sweep = sweeps[-1]
            sweeps = sweeps[:-1]
        else:
            buffered_sweep = None

        code_path = Path(os.path.realpath(__file__)).parent
        code_archive = {}
        for file in os.listdir(code_path):
            try:
                with open(f"{code_path}/{file}", "r") as f:
                    code_archive[file] = f.read()
            except Exception:
                pass

        try:
            if self.fridge_name:
                self.cryostat = self.fridge_name
            else:
                self.cryostat = socket.gethostname().split(".")[0].split("-")[1]
        except Exception:
            self.cryostat = "dummy"

        meta = {
            "Timestamp": datetime.datetime.now().isoformat(),
            "Cryostat": self.cryostat,
            "Measurement ID": self.id,
            "Wafer ID": self.wafer_id,
            "Device Type": self.device_type,
            "Sample Name": self.sample_name,
            "Experiment Name": self.experiment,
            "Requirements": self.get_installed_packages(),
            # "Code Archive": str(code_archive),
        }

        data_vars = {
            f"{dependent.name}": self._make_dataarray(sweeps, dependent)
            for dependent in dependents
        }

        coords = {
            f"{param.name}": sweep.values
            for sweep in sweeps
            for param in sweep.parameter
        }

        # Handle buffered dependents
        if buffered_sweep:
            for buffered_dependent in self.buffered_dependents_tree:
                sweeps_full = (
                    sweeps + self.buffered_dependents_tree[buffered_dependent]["sweeps"]
                )
                data_vars[buffered_dependent.name] = self._make_dataarray(
                    sweeps_full, buffered_dependent
                )

                for sweep in sweeps_full:
                    for param in sweep.parameter:
                        coords[param.name] = sweep.values
        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=meta)

        for sweep in sweeps:
            for param in sweep.parameter:
                ds.coords[param.name].attrs["instrument"] = param.instrument.name
                ds.coords[param.name].attrs["unit"] = param.unit
                ds.coords[param.name].attrs["label"] = param.label
        return ds

    def _push_gitlab(self, dataset, data_hash):
        """
        Record a measurement hash and metadata in the remote hash repository.

        The record is committed on the cryostat branch and pushed to the
        configured Git repository. The measurement data itself is not pushed.

        Args:
            dataset (xr.Dataset): Final dataset whose attributes are recorded.
            data_hash (str): Digest of the persisted measurement data.

        """
        git_ssh_identity_file = str(Path.home() / ".ssh" / "id_rsa")
        known_hosts = str(Path.home() / ".ssh" / "known_hosts")

        git_ssh_cmd = (
            f"ssh -i {git_ssh_identity_file} "
            f"-o StrictHostKeyChecking=accept-new "
            f"-o UserKnownHostsFile={known_hosts}"
        )

        repo_path = Path(self.git_repo)

        if not (repo_path / ".git").exists():
            logger.info("Cloning the measurement-hashes repository")

            from git import Repo

            Repo.clone_from(
                url="git@git.pgi.fz-juelich.de:squad-lab/hashes.git",
                to_path=str(repo_path),
                single_branch=True,
                branch="main",
                env=dict(GIT_SSH_COMMAND=git_ssh_cmd),
            )

        # Imported here, not at module scope: GitPython refuses to
        # initialise without a git executable on PATH, which would make git a
        # hard requirement of `import qanary.measure` for every user.
        from git import Repo

        repo = Repo(str(repo_path))

        with repo.git.custom_environment(GIT_SSH_COMMAND=git_ssh_cmd):
            repo.git.fetch("origin")

            local_branches = [h.name for h in repo.heads]
            remote_branches = repo.git.branch("-r").splitlines()

            if self.cryostat in local_branches:
                logger.info(f"Checking out branch: {self.cryostat}")
                repo.git.checkout(self.cryostat)
            elif any(f"origin/{self.cryostat}" in rb for rb in remote_branches):
                logger.info(f"Checking out tracking branch: {self.cryostat}")
                repo.git.checkout("-b", self.cryostat, f"origin/{self.cryostat}")
            else:
                logger.info(f"Creating new branch: {self.cryostat}")
                repo.git.checkout("-b", self.cryostat)

            hash_location = (
                repo_path
                / self.wafer_id
                / self.device_type
                / self.sample_name
                / self.experiment
            )
            hash_location.mkdir(parents=True, exist_ok=True)

            out_file = hash_location / str(self.id)
            out_file.write_text(
                f"Hash: {data_hash}\n\n{json.dumps(dataset.attrs, indent=2)}"
            )

            repo.git.add(all=True)

            if repo.is_dirty(untracked_files=True):
                repo.git.commit("-m", f"Add new measurement hash: {self.id}")
                repo.git.push("--set-upstream", "origin", self.cryostat)
            else:
                logger.info("No changes to commit; skipping push.")

    def _compute_data_hash(self) -> str:
        """
        Compute the SHA-256 digest of the persisted measurement output.

        Legacy directory outputs are hashed recursively; current netCDF outputs
        are read in chunks.

        Returns:
            str: Hexadecimal SHA-256 digest.

        """
        data_path = Path(self.data)
        if data_path.is_dir():
            return dirhash(str(data_path), "sha256", excluded_extensions=["pyc"])

        digest = hashlib.sha256()
        with open(data_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)

        return digest.hexdigest()

    def _rescue_live_store(self) -> bool:
        """
        Preserve a live store whose measurement failed to export.

        Returns:
            bool: Whether the conversion produced ``self.data`` after all, so
                the caller can treat the export as having succeeded.

        """
        store = Path(self.live_data)
        if not store.exists():
            logger.error(
                f"Measurement {self.id} exported nothing and has no live store "
                "to fall back on. Its data is lost."
            )
            return False

        rescued = Path(self.datalogging) / store.name
        try:
            shutil.move(str(store), str(rescued))
            _target_marker(store).unlink(missing_ok=True)
        except Exception as exc:
            logger.error(
                f"Measurement {self.id} failed its netCDF export and its live "
                f"store could not be moved out of {store}: {exc}. Recover it by "
                "hand -- the next run prunes stores left there."
            )
            return False

        try:
            _convert_live_store(self.data, overwrite=True)
        except Exception as exc:
            logger.error(
                f"Measurement {self.id} failed its netCDF export, and the live "
                f"store kept at {rescued} could not be converted either: {exc}. "
                "Convert it with `qanary.dataset.convert` once the cause is "
                "fixed. It is very likely incomplete."
            )
            return False

        logger.error(
            f"Measurement {self.id} failed its netCDF export and was recovered "
            f"from its live store into {self.data}. It is very likely "
            "incomplete."
        )
        return True

    def _finalize_disk_artifacts(
        self, dataset: Optional[xr.Dataset] = None, verbose: bool = False
    ) -> Optional[xr.Dataset]:
        """
        Export the latest dataset to netCDF and remove temporary live artifacts.

        The supplied dataset is preferred, followed by the current in-memory
        dataset and then the disk checkpoint. Export uses a temporary file and
        atomic replacement. If that fails, the live Zarr store is moved beside
        the intended target and converted there. Successful export updates the
        path advertised by Qimchi Connect before removing the live store and its
        recovery marker.

        Args:
            dataset (Optional[xr.Dataset]): Dataset to persist. If omitted, use
                the newest available measurement state.
            verbose (bool): Log successful export details. Defaults to False.

        Returns:
            Optional[xr.Dataset]: Exported dataset, or ``None`` if no export
                could be completed.

        """
        # self.arr is the authoritative, most recent in-memory state during a
        # run. The disk Zarr checkpoint can lag behind after a transient lock.
        export_dataset = dataset if dataset is not None else self.arr

        if export_dataset is None:
            try:
                if Path(self.live_data).exists():
                    export_dataset = xr.load_dataset(self.live_data, engine="zarr")
            except Exception as e:
                logger.warning(
                    f"Failed loading latest dataset state from {self.live_data}: {e}"
                )

        if export_dataset is None and self.arr is not None:
            export_dataset = self.arr

        if export_dataset is None:
            logger.warning(
                "No dataset available to export to netCDF during finalization"
            )
            return None

        nc_exported = False
        try:
            tmp_nc = f"{self.data}.tmp"
            export_dataset.to_netcdf(tmp_nc, mode="w")
            os.replace(tmp_nc, self.data)
            nc_exported = True
            if verbose:
                logger.info(f"[measurement] Exported final dataset to {self.data}")
        except Exception as e:
            logger.error(f"Failed exporting final dataset to netCDF: {e}")
            nc_exported = self._rescue_live_store()

        if nc_exported:
            nc_path = str(Path(self.data).resolve())
            try:
                update_live_disk_path(self.id, nc_path)
            except Exception as e:
                logger.warning(f"Failed updating the advertised measurement path: {e}")

            try:
                if Path(self.live_data).exists():
                    shutil.rmtree(self.live_data)
                _target_marker(Path(self.live_data)).unlink(missing_ok=True)
                reset_disk_persist_cache(self.live_data)
            except Exception as e:
                logger.warning(
                    f"Failed removing temporary zarr store at {self.live_data}: {e}"
                )

        if not nc_exported:
            return None

        return export_dataset

    def _print_table(self, snapshot_table, headers):
        """
        Print aligned measurement metadata rows.

        Args:
            snapshot_table (list[list]): Rows to print.
            headers (list[str]): Column headings.

        """

        # Convert everything to string
        table = [[str(cell) for cell in row] for row in snapshot_table]

        # Compute column widths
        ncols = len(headers)
        col_widths = []
        for col in range(ncols):
            max_cell = max(len(row[col]) for row in table)
            col_widths.append(max(max_cell, len(headers[col])))

        # Helpers
        def fmt_row(row):
            return "  ".join(row[i].ljust(col_widths[i]) for i in range(ncols))

        def sep():
            return "  ".join("-" * w for w in col_widths)

        # Print table
        print(fmt_row(headers))
        print(sep())
        for row in table:
            print(fmt_row(row))

    def run(
        self,
        # NOTE: PointSweep is structurally compatible at runtime even though
        # this legacy annotation names only Sweep and CircularSweep.
        sweeps: Union[Sweep, CircularSweep, Sequence[Union[Sweep, dict]]],
        dependents: list,
        interrupt: Callable = lambda: False,
        rampdown_on_interrupt=False,
        verbose: bool = False,
        no_hashing: bool = False,
    ):
        """
        Acquire a measurement and finalize it to netCDF.

        Args:
            sweeps (Sweep | CircularSweep | Sequence[Sweep | dict]): A sweep,
                nested slow sweeps, or slow sweeps followed by one buffered-tree
                mapping.
            dependents (list): QCoDeS parameters read at every unbuffered point.
                Buffered dependents belong in the buffered-tree mapping instead.
            interrupt (Callable): Callback checked during acquisition. A truthy
                result interrupts the run. Defaults to always false.
            rampdown_on_interrupt (bool): Ramp swept parameters to zero after a
                ``KeyboardInterrupt``. Defaults to False.
            verbose (bool): Log checkpoint operations. Defaults to False.
            no_hashing (bool): Skip recording the completed file's hash in the
                measurement-hash repository. Defaults to False.

        Raises:
            InterruptedError: If the *interrupt* callback requests a stop.
            KeyboardInterrupt: If the user interrupts acquisition.
            RuntimeError: If acquisition returns without visiting every
                expected point.

        Notes:
            Partial data is finalized on interruption or failure before the
            original exception is re-raised. The method returns ``None``; use
            :attr:`data` for the completed file path.

        """
        if not isinstance(sweeps, Sequence):
            sweeps = [sweeps]

        sweeps_complete = sweeps
        if isinstance(sweeps[-1], dict):
            buffered_sweep = sweeps[-1]
            sweeps = sweeps[:-1]
        else:
            buffered_sweep = None

        # Get instrument snapshots for metadata
        independents = []
        instruments_snapshot = {
            inst.name: inst.snapshot() for inst in self.station.instruments
        }

        parameters_snapshot = {}

        for param in self.station.parameters:
            val = param()

            if isinstance(val, np.ndarray):
                val = val.tolist()

            val_str = str(val)
            if len(val_str) > 50:
                val_str = val_str[:50] + "..."

            parameters_snapshot[param.name] = {
                "value": val_str,
                "unit": param.unit,
                "label": param.label,
            }

        # Get sweep metadata for pretty printed table
        def _fmt_range(values, ndp: int = 3) -> str:
            return f"{values[0]:.{ndp}e} to {values[-1]:.{ndp}e}"

        sweep_metadata = []
        sweep_metadata_headers = [
            "Independent(s)",
            "Range",
            "Number of Points",
            "Delay (s)",
        ]

        sweep_dims = 0
        for sweep in sweeps:
            sweep_metadata.append(
                [
                    ",".join([param.label for param in sweep.parameter]),
                    _fmt_range(sweep.values),
                    len(sweep.values),
                    sweep.delay,
                ]
            )
            for param in sweep.parameter:
                independents.append(param)
                sweep_dims += 1

        if buffered_sweep:
            self.buffered_dependents_tree = {}
            _fetch_dependents_tree(buffered_sweep, state=self.buffered_dependents_tree)
            self.buffered_dependents_tree = self.buffered_dependents_tree[
                "dependent_tree"
            ]

            seen = set()
            for buffered_dependent in self.buffered_dependents_tree:
                for sweep in self.buffered_dependents_tree[buffered_dependent][
                    "sweeps"
                ]:
                    if id(sweep) in seen:
                        continue
                    seen.add(id(sweep))

                    sweep_metadata.append(
                        [
                            ",".join([param.label for param in sweep.parameter]),
                            _fmt_range(sweep.values),
                            len(sweep.values),
                            sweep.delay,
                        ]
                    )
                    for param in sweep.parameter:
                        independents.append(param)
                        sweep_dims += 1

        parameters_snapshot_headers = ["Name", "Label", "Value", "Unit"]
        parameters_snapshot_table = [
            [
                param.name,
                parameters_snapshot[param.name]["label"],
                parameters_snapshot[param.name]["value"],
                parameters_snapshot[param.name]["unit"],
            ]
            for param in self.station.parameters
        ]

        swm_list = [dict(zip(sweep_metadata_headers, swm)) for swm in sweep_metadata]

        try:
            inst_snap_json = json.dumps(instruments_snapshot)
        except TypeError:
            inst_snap_json = json.dumps(_sanitize_for_json(instruments_snapshot))

        meta = {
            "Instruments Snapshot": inst_snap_json,
            "Parameters Snapshot": json.dumps(parameters_snapshot),
            "Sweeps": json.dumps({swm["Independent(s)"]: swm for swm in swm_list}),
            "Extra Metadata": json.dumps(self.extra_metadata),
        }

        # make empty dataset with global dimensions and buffered dimensions
        self.arr = self._make_dataset(sweeps_complete, dependents)

        self.arr.attrs.update(meta)
        self.memory_store = zarr.MemoryStore()
        self.disk_store = zarr.DirectoryStore(self.live_data)
        self.arr.to_zarr(store=self.memory_store, mode="w")
        if verbose:
            logger.info("[measurement] Seeded dataset to in-memory store")
        zarr.copy_store(self.memory_store, self.disk_store, if_exists="replace")
        # Checkpoints track what they have already written; this seeding copy
        # bypasses that, so start the run from a known-empty cache.
        reset_disk_persist_cache(self.disk_store)
        if verbose:
            logger.info("[measurement] Seeded dataset to disk .zarr store")
        _register_memory_store(
            self.id, self.memory_store, disk_path=str(Path(self.live_data).resolve())
        )
        logger.debug(f"Live Memory Location: memory://{self.id}")

        # Do the measurement
        dataset: Optional[xr.Dataset] = None
        finalized = False
        try:
            total_points = 1
            for sweep in sweeps:
                total_points *= len(sweep.values)
            if buffered_sweep is not None:
                buffered_points, _ = _buffered_sweep_progress_info(buffered_sweep)
                total_points *= buffered_points

            logger.info("Registered Parameters:")
            print("\n")
            self._print_table(
                snapshot_table=parameters_snapshot_table,
                headers=parameters_snapshot_headers,
            )
            print("\n")
            logger.info("Sweeps Summary:")
            print("\n")
            self._print_table(
                snapshot_table=sweep_metadata,
                headers=sweep_metadata_headers,
            )
            print("\n")

            logger.info(f"Starting the measurement with ID: {self.id}")
            global bar
            bar = tqdm(
                total=total_points,
                ascii="*ᗧⵔ•",
                ncols=10,
                dynamic_ncols=True,
                desc="Measurement Progress",
                unit="point",
            )

            dataset = _stepper(
                dataset=self.arr,
                data_location=self.live_data,
                depth=len(sweeps),
                sweeps=sweeps,
                independents=independents,
                dependents=dependents,
                sweep_cache=[0.0] * sweep_dims,
                bar=bar,
                save_interval=self.save_interval,
                interrupt=interrupt,
                memory_store=self.memory_store,
                disk_store=self.disk_store,
                nc_snapshot_path=self.data if self.nc_snapshot_during_run else None,
                verbose=verbose,
                buffered_sweep=buffered_sweep,
            )

            completed_points = int(bar.n)
            expected_points = int(bar.total)
            if completed_points != expected_points:
                raise RuntimeError(
                    "Measurement returned before all sweep points completed: "
                    f"{completed_points}/{expected_points} points acquired"
                )

            dataset = self._finalize_disk_artifacts(dataset=dataset, verbose=verbose)
            finalized = True
            if dataset is None:
                logger.warning("Measurement finished but final netCDF export failed")
                return

            if not no_hashing:
                data_hash = self._compute_data_hash()
                try:
                    self._push_gitlab(dataset, data_hash)
                    logger.info(
                        f"Measurement completed and pushed with hash: {data_hash}"
                    )
                except Exception as e:
                    logger.error(
                        f"Did not push measurement hash to gitlab (upstream). Will try again after next measurement: {e}"
                    )

            return

        except KeyboardInterrupt:
            logger.warning("Measurement interrupted; stopping buffered instruments")
            if buffered_sweep is not None:
                _abort_instruments(buffered_sweep)

            if rampdown_on_interrupt:
                logger.info("Ramping down swept parameters")
                rampdown_sweeps = [
                    Sweep(parameter, parameter(), 0.0, num=100, delay=1e-2)
                    for sweep in sweeps
                    for parameter in _sweep_parameters(sweep)
                ]
                buffered_tree = getattr(self, "buffered_dependents_tree", None) or {}
                for entry in buffered_tree.values():
                    for sw in entry["sweeps"]:
                        for parameter in _sweep_parameters(sw):
                            rampdown_sweeps.append(
                                Sweep(
                                    parameter,
                                    parameter(),
                                    0.0,
                                    num=100,
                                    delay=1e-2,
                                )
                            )
                sweeper(rampdown_sweeps)
            raise
        except Exception as e:
            logger.exception(f"Measurement failed with error: {e}", exc_info=True)
            raise
        finally:
            if bar is not None:
                try:
                    bar.close()
                except Exception:
                    pass
            if not finalized:
                self._finalize_disk_artifacts(dataset=dataset, verbose=verbose)
            _unregister_memory_store(self.id)


def run(
    # NOTE: PointSweep is structurally compatible at runtime; see Measurement.run.
    sweeps: Union[Sweep, CircularSweep, Sequence[Sweep]],
    dependents: list,
    wafer_id: str,
    device_type: str,
    sample_name: str,
    experiment_name: str,
    metadata: dict = {},
    station: Station = None,
    data_location: str = "./test/",
    interrupt: Callable = lambda: None,
    rampdown_on_interrupt=False,
    location_return=False,
    verbose: bool = False,
    no_hashing: bool = False,
    *args,
    **kwargs,
):
    """
    Create and run a :class:`Measurement` in one call.

    Args:
        sweeps (Sweep | CircularSweep | Sequence[Sweep | dict]): Sweep
            definition accepted by :meth:`Measurement.run`.
        dependents (list): Unbuffered QCoDeS parameters to record.
        wafer_id (str): Wafer identifier used in the output path.
        device_type (str): Device type used in the output path.
        sample_name (str): Sample name used in the output path.
        experiment_name (str): Experiment name used in the output path.
        metadata (dict): Extra metadata stored on the dataset.
        station (Station): Instruments and parameters to snapshot.
        data_location (str): Root directory for completed measurements.
            Defaults to ``"./test/"``.
        interrupt (Callable): Callback checked during acquisition.
        rampdown_on_interrupt (bool): Ramp swept parameters to zero after a
            ``KeyboardInterrupt``. Defaults to False.
        location_return (bool): Return the output path after the run. Defaults
            to False.
        verbose (bool): Log checkpoint operations. Defaults to False.
        no_hashing (bool): Skip recording the completed file's hash. Defaults
            to False.
        *args: Additional positional arguments forwarded to
            :class:`Measurement`.
        **kwargs: Additional keyword arguments forwarded to
            :class:`Measurement`.

    Returns:
        str | None: Final netCDF path when *location_return* is true; otherwise
            ``None``.

    Raises:
        ValueError: If *station* is not provided.

    """
    if not station:
        raise ValueError("Station is required")

    meas = Measurement(
        sample_name=sample_name,
        wafer_id=wafer_id,
        device_type=device_type,
        experiment_name=experiment_name,
        station=station,
        data_location=data_location,
        metadata=metadata,
        *args,
        **kwargs,
    )

    meas.run(
        sweeps,
        dependents,
        interrupt,
        rampdown_on_interrupt,
        verbose=verbose,
        no_hashing=no_hashing,
    )
    if location_return:
        return meas.data
