import datetime
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
import xarray as xr
import zarr
from checksumdir import dirhash
from git import Repo
from qcodes.parameters import Parameter
from tqdm import tqdm

# Local imports
from qcutils import live_db, live_server
from qcutils.buffered.sweep import fetch_dependents_tree
from qcutils.logger import get_logger
from qcutils.sweep import CircularSweep, Sweep, stepper, sweeper
from qcutils.parameters import ParameterMixin

logger = get_logger(__name__)


@dataclass
class LiveDatasetInfo:
    store: Optional[zarr.MemoryStore]
    disk_path: Optional[str] = None
    host: str = field(default_factory=lambda: socket.gethostname())
    pid: int = field(default_factory=os.getpid)
    started_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


# Registry tracking active in-memory Zarr stores keyed by measurement ID.
LIVE_MEMORY_STORES: Dict[str, Union[zarr.MemoryStore, LiveDatasetInfo]] = {}


def register_memory_store(
    measurement_id: str, store: zarr.MemoryStore, disk_path: Optional[str] = None
) -> int:
    """
    Register a memory store and start WebSocket server.

    Returns:
        int: WebSocket port number if server started, 0 otherwise
    """
    info = LiveDatasetInfo(store=store, disk_path=disk_path)
    LIVE_MEMORY_STORES[measurement_id] = info

    ws_port = 0

    # Start WebSocket server if not already running
    if live_server.is_server_running():
        ws_port = live_server.get_server_port()
        logger.debug(
            f"WebSocket server already running on port {ws_port}, registered measurement {measurement_id}"
        )
    else:
        logger.info("Starting WebSocket server for live data...")
        # NOTE: Find available port starting from 8765
        ws_port = _find_available_port(8765)
        live_server.set_memory_stores_reference(LIVE_MEMORY_STORES)
        if live_server.start_live_server(port=ws_port):
            logger.info(
                f"Live data WebSocket server started on ws://localhost:{ws_port}"
            )
        else:
            logger.error("Failed to start live data WebSocket server")
            ws_port = 0

    # Register in database if available
    if live_db and ws_port > 0 and disk_path:
        try:
            live_db.init_database()
            ws_url = f"ws://localhost:{ws_port}"
            started_at = datetime.datetime.utcnow().isoformat()
            live_db.register_measurement(
                measurement_id=measurement_id,
                fpath=disk_path,
                ws_url=ws_url,
                ws_port=ws_port,
                started_at=started_at,
            )
            logger.info(f"Registered measurement {measurement_id} in database")
        except Exception as e:
            logger.warning(f"Failed to register measurement in database: {e}")

    return ws_port


def _find_available_port(start_port: int, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                return port
            except OSError:
                continue
    # Fallback to start_port if none found
    return start_port


def get_memory_store(measurement_id: str) -> Optional[zarr.MemoryStore]:
    entry = LIVE_MEMORY_STORES.get(measurement_id)
    if isinstance(entry, LiveDatasetInfo):
        return entry.store
    return entry


def unregister_memory_store(measurement_id: str) -> None:
    LIVE_MEMORY_STORES.pop(measurement_id, None)

    # Mark as ended in database
    if live_db:
        try:
            ended_at = datetime.datetime.utcnow().isoformat()
            live_db.end_measurement(measurement_id, ended_at)
            logger.info(f"Marked measurement {measurement_id} as ended in database")
        except Exception as e:
            logger.warning(f"Failed to mark measurement as ended in database: {e}")


def get_live_dataset_info(measurement_id: str) -> Optional[LiveDatasetInfo]:
    """Return LiveDatasetInfo for the given measurement if available."""

    entry = LIVE_MEMORY_STORES.get(measurement_id)
    if isinstance(entry, LiveDatasetInfo):
        return entry
    if entry is not None:
        return LiveDatasetInfo(store=entry)

    return None


def list_live_measurements(include_registry: bool = True) -> Dict[str, LiveDatasetInfo]:
    """Return mapping of measurement id to LiveDatasetInfo."""

    results: Dict[str, LiveDatasetInfo] = {}
    for measurement_id, entry in LIVE_MEMORY_STORES.items():
        if isinstance(entry, LiveDatasetInfo):
            results[measurement_id] = entry
        else:
            results[measurement_id] = LiveDatasetInfo(store=entry)

    return results


bar = None


def _sanitize_for_json(obj):
    """
    Recursively convert numpy arrays in a structure to lists,
    so the result is JSON-serializable.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]

    return obj


class Station:
    def __init__(self, name: str):
        self.name = name
        self.instruments = []
        self.parameters = []

    def add_parameter(
        self, name: str, label: str, param: Parameter, param_type: str = "gate"
    ):
        pm = ParameterMixin(param, name, label, param_type)
        param.label = label

        if pm in self.parameters:
            raise ValueError(
                f"Parameter {pm.name} already exists in station {self.name}"
            )
        else:
            self.parameters.append(pm)
        return pm

    def remove_parameter(self, pm: ParameterMixin):
        if pm in self.parameters:
            self.parameters.remove(pm)
        else:
            raise ValueError(f"Parameter {pm.name} not found in station {self.name}")


class Measurement:
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
        """Measurement class for sweeping and storing data

        Args:
            wafer_id (str): ID of the wafer
            device_type (str): Type of the device
            sample_name (str): Name of the sample
            experiment_name (str): Name of the experiment
            data_location (str): Location of the data to be stored
            save_interval (float, optional): Period for writing to the disk. Defaults to 0.1.
            nc_snapshot_during_run (bool, optional): Persist best-effort .nc snapshots during run. Defaults to True.
            git_repo (str, optional): Location of the git repository to store the measurement hashes. Defaults to "~/.measurement-hashes".
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
        self.live_data = f"{self.datalogging}/{self.id}.zarr"
        self.arr = None
        self.memory_store = None
        self.disk_store = None
        self.station = station
        self.fridge_name = fridge_name
        logger.info(f"Measurement Location: {self.data}")

    def get_installed_packages(self):
        try:
            pip_output = subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"]
            ).decode()
        except subprocess.CalledProcessError:
            pip_output = ""

        try:
            uv_output = subprocess.check_output(["uv", "pip", "freeze"]).decode()
        except subprocess.CalledProcessError:
            uv_output = ""

        return pip_output + uv_output

    def _make_dataarray(self, sweeps, dependent):
        """Create a dataarray for the dependent parameter

        Args:
            sweeps (Sequence[Sweep]): List of sweep objects
            dependent (Parameter): QCoDeS parameter

        Returns:
            xr.DataArray: DataArray for the dependent parameter
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
        """Create an xarray dataset for the measurement

        Args:
            sweeps (Sequence[Sweep]): List of sweep objects
            dependents (list): List of dependent QCoDeS parameters
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
        """Push the data to the gitlab repository"""
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

            Repo.clone_from(
                url="git@git.pgi.fz-juelich.de:squad-lab/hashes.git",
                to_path=str(repo_path),
                single_branch=True,
                branch="main",
                env=dict(GIT_SSH_COMMAND=git_ssh_cmd),
            )

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
        Util function to compute SHA256 hash for the persisted measurement output.

        """
        data_path = Path(self.data)
        if data_path.is_dir():
            return dirhash(str(data_path), "sha256", excluded_extensions=["pyc"])

        digest = hashlib.sha256()
        with open(data_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)

        return digest.hexdigest()

    def _finalize_disk_artifacts(
        self, dataset: Optional[xr.Dataset] = None, verbose: bool = False
    ) -> Optional[xr.Dataset]:
        """
        Best-effort finalize to a single netCDF file and update live DB path.
        This is called on normal completion and interruption/error paths.

        Args:
            dataset (Optional[xr.Dataset]): The dataset to persist. If None, will attempt to load from live Zarr or in-memory store.
            verbose (bool): Whether to log detailed information during finalization.

        Returns:
            Optional[xr.Dataset]: The dataset that was finalized, or None if finalization failed.

        Raises:
            Exception: If there is an error during finalization, it will be logged but not raised.

        """
        export_dataset = dataset

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

        if nc_exported:
            nc_path = str(Path(self.data).resolve())
            if live_db:
                try:
                    live_db.update_measurement_path(self.id, nc_path)
                except Exception as e:
                    logger.warning(f"Failed updating final measurement path in DB: {e}")

            entry = LIVE_MEMORY_STORES.get(self.id)
            if isinstance(entry, LiveDatasetInfo):
                entry.disk_path = nc_path

            try:
                if Path(self.live_data).exists():
                    shutil.rmtree(self.live_data)
            except Exception as e:
                logger.warning(
                    f"Failed removing temporary zarr store at {self.live_data}: {e}"
                )

        return export_dataset

    def _print_table(self, snapshot_table, headers):
        """
        Print out a parameter snapshot table in a tabular shape.

        Parameters
        ----------
        snapshot_table : list[list[str]]
            Table rows, e.g. [[up1, Up Plunger 1, -1.0, V], ...]
        headers : list[str]
            Column headers, e.g. ["Name", "Label", "Value", "Unit"]
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
        sweeps: Union[Sweep, CircularSweep, Sequence[Union[Sweep, dict]]],
        dependents: list,
        interrupt: Callable = lambda: False,
        rampdown_on_interrupt=False,
        verbose: bool = False,
    ):
        """Run the measurement

        Args:
            sweeps (Union[Sweep, CircularSweep, Sequence[Sweep]]): Sweep object or list of sweep objects
            dependents (list): List of dependent QCoDeS parameters
            interrupt (Callable, optional): Function to interrupt the measurement. Defaults to None.
            rampdown_on_interrupt (bool, optional): Ramp down the instruments on interrupt. Defaults to False.
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
            fetch_dependents_tree(buffered_sweep, state=self.buffered_dependents_tree)
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
        if verbose:
            logger.info("[measurement] Seeded dataset to disk .zarr store")
        register_memory_store(
            self.id, self.memory_store, disk_path=str(Path(self.live_data).resolve())
        )
        logger.debug(f"Live Memory Location: memory://{self.id}")

        # Do the measurement
        dataset: Optional[xr.Dataset] = None
        finalized = False
        interrupted = False
        return_code = None
        try:
            total_points = 1
            for sweep in sweeps:
                total_points *= len(sweep.values)

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
            )

            dataset = stepper(
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

            dataset = self._finalize_disk_artifacts(dataset=dataset, verbose=verbose)
            finalized = True
            if dataset is None:
                logger.warning("Measurement finished but final netCDF export failed")
                return
            data_hash = self._compute_data_hash()
            try:
                self._push_gitlab(dataset, data_hash)
                logger.info(f"Measurement completed and pushed with hash: {data_hash}")
            except Exception as e:
                logger.error(
                    f"Did not push measurement hash to gitlab (upstream). Will try again after next measurement: {e}"
                )

            return

        except KeyboardInterrupt:
            interrupted = True
            logger.warning("Measurement interrupted, ramping down instruments")
            if rampdown_on_interrupt:
                rampdown_sweeps = [
                    Sweep(sweep.parameter, sweep.parameter(), 0.0, num=100, delay=1e-2)
                    for sweep in sweeps
                ]
                for sweep in self.buffered_dependents_tree:
                    for sw in self.buffered_dependents_tree[sweep]["sweeps"]:
                        rampdown_sweeps.append(
                            Sweep(
                                sw.parameter, sw.parameter(), 0.0, num=100, delay=1e-2
                            )
                        )
                sweeper(rampdown_sweeps)
                return_code = 1
        except Exception as e:
            logger.exception(f"Measurement failed with error: {e}", exc_info=True)
        finally:
            if bar is not None:
                try:
                    bar.close()
                except Exception:
                    pass
            if not finalized:
                self._finalize_disk_artifacts(dataset=dataset, verbose=verbose)
            unregister_memory_store(self.id)

        if interrupted:
            return return_code


def run(
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
    *args,
    **kwargs,
):
    """Helper function to run a measurement, refer to the Measuremment class for more details"""
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

    meas.run(sweeps, dependents, interrupt, rampdown_on_interrupt, verbose=verbose)
    if location_return:
        return meas.data
