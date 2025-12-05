import datetime
import json
import os
import socket
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
import xarray as xr
import zarr
from checksumdir import dirhash
from git import GitCommandError, Repo
from qcodes.parameters import Parameter
from tabulate import tabulate
from tqdm import tqdm

# Local imports
from qcutils import live_db, live_server
from qcutils.logger import get_logger
from qcutils.sweep import CircularSweep, Sweep, stepper, sweeper
from qcutils.buffered.sweep import fetch_dependents_tree

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


class ParameterMixin:
    def __init__(
        self,
        param: Parameter,
        name: str = None,
        label: str = None,
        param_type: str = "gate",
    ):
        self.__dict__.update(param.__dict__)
        self.__class__ = param.__class__

        self.param_type = param_type
        if name:
            self._short_name = name
        if label:
            self._label = label


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
        save_interval: float = 0.1,
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
        self.git_repo = os.path.expanduser(git_repo)

        data_files = os.listdir(self.datalogging)
        data_files = [
            int(file.split("-")[0]) for file in data_files if file.endswith(".zarr")
        ]

        if len(data_files) != 0:
            self.id = f"{sorted(data_files)[-1] + 1}-{uuid.uuid4()}"

        self.data = f"{self.datalogging}/{self.id}.zarr"
        self.arr = None
        self.memory_store = None
        self.disk_store = None
        self.station = station
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
            cryostat_name = socket.gethostname().split(".")[0].split("-")[1]
        except Exception:
            cryostat_name = "dummy"

        meta = {
            "Timestamp": datetime.datetime.now().isoformat(),
            "Cryostat": cryostat_name,
            "Measurement ID": self.id,
            "Wafer ID": self.wafer_id,
            "Device Type": self.device_type,
            "Sample Name": self.sample_name,
            "Experiment Name": self.experiment,
            "Requirements": self.get_installed_packages(),
            # "Code Archive": str(code_archive),
        }

        self.cryostat = meta["Cryostat"]

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
                sweeps_full = sweeps + self.buffered_dependents_tree[buffered_dependent]["sweeps"]
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
        """Push the data to the gitlab repository

        Args:
            dataset (xarray.Dataset): xarray dataset
            data_hash (str): Hash of the data
        """
        git_ssh_identity_file = str(PurePosixPath(Path.home() / ".ssh" / "id_rsa"))
        git_ssh_cmd = f"ssh -i {git_ssh_identity_file}"

        if not os.path.exists(Path("~/.measurement-hashes/.git").expanduser()):
            logger.info("Cloning the measurement-hashes repository")
            Repo.clone_from(
                url="git@git.pgi.fz-juelich.de:squad-lab/hashes.git",
                to_path=self.git_repo,
                single_branch=True,
                branch=self.cryostat,
                env=dict(GIT_SSH_COMMAND=git_ssh_cmd),
            )
        repo = Repo(self.git_repo)

        if f"{self.cryostat}" not in repo.git.branch().split("* ")[1].split("\n"):
            logger.info(f"Creating and checking out to branch: {self.cryostat}")
            try:
                repo.git.checkout(b=f"{self.cryostat}")
            except GitCommandError:
                repo.git.branch(d=f"{self.cryostat}")
                repo.git.checkout(b=f"{self.cryostat}")
        else:
            logger.info(f"Checking out to branch: {self.cryostat}")
            repo.git.checkout(f"{self.cryostat}")

        hash_location = f"{self.git_repo}/{self.wafer_id}/{self.device_type}/{self.sample_name}/{self.experiment}"
        os.makedirs(hash_location, exist_ok=True)
        with open(
            f"{hash_location}/{self.id}",
            "w",
        ) as f:
            f.write(f"Hash: {data_hash}\n\n{json.dumps(dataset.attrs, indent=2)}")
        logger.info("Adding the measurement hash to the git repository")
        repo.git.add(all=True)
        repo.git.commit("-m", f"Add new measurement hash: {self.id}")
        repo.git.push("--set-upstream", "origin", f"{self.cryostat}")

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
        parameters_snapshot = {
            param.name: {
                "value": param(),
                "unit": param.unit,
                "label": param.label,
            }
            for param in self.station.parameters
        }

        # Get sweep metadata for pretty printed table
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
                    f"{sweep.values[0]} to {sweep.values[-1]}",
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
            # original state consists of sweeps, their dims and everything. We only need the dependent_tree
            self.buffered_dependents_tree = self.buffered_dependents_tree["dependent_tree"]
            for buffered_dependent in self.buffered_dependents_tree:
                for sweep in self.buffered_dependents_tree[buffered_dependent][
                    "sweeps"
                ]:
                    sweep_metadata.append(
                        [
                            ",".join([param.label for param in sweep.parameter]),
                            f"{sweep.values[0]} to {sweep.values[-1]}",
                            len(sweep.values),
                            sweep.delay,
                        ]
                    )
                    for param in sweep.parameter:
                        independents.append(param)
                        sweep_dims += 1

        swm_list = [dict(zip(sweep_metadata_headers, swm)) for swm in sweep_metadata]
        meta = {
            "Instruments Snapshot": json.dumps(instruments_snapshot),
            "Parameters Snapshot": json.dumps(parameters_snapshot),
            "Sweeps": json.dumps({swm["Independent(s)"]: swm for swm in swm_list}),
            "Extra Metadata": json.dumps(self.extra_metadata),
        }
        
        # make empty dataset with global dimensions and buffered dimensions
        self.arr = self._make_dataset(sweeps_complete, dependents)
        
        self.arr.attrs.update(meta)
        self.memory_store = zarr.MemoryStore()
        self.disk_store = zarr.DirectoryStore(self.data)
        self.arr.to_zarr(store=self.memory_store, mode="w")
        if verbose:
            logger.info("[measurement] Seeded dataset to in-memory store")
        zarr.copy_store(self.memory_store, self.disk_store, if_exists="replace")
        if verbose:
            logger.info("[measurement] Seeded dataset to disk store")
        register_memory_store(self.id, self.memory_store, disk_path=self.data)
        logger.debug(f"Live Memory Location: memory://{self.id}")

        # Do the measurement
        try:
            total_points = 1
            for sweep in sweeps:
                total_points *= len(sweep.values)

            print(
                "\n",
                tabulate(
                    sweep_metadata,
                    headers=sweep_metadata_headers,
                ),
                "\n",
            )

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
                data_location=self.data,
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
                verbose=verbose,
                buffered_sweep=buffered_sweep,
            )
            bar.close()

            data_hash = dirhash(self.data, "sha256", excluded_extensions=["pyc"])
            try:
                self._push_gitlab(dataset, data_hash)
                logger.info(f"Measurement completed and pushed with hash: {data_hash}")
            except Exception as e:
                logger.error(f"Did not push measurement hash to gitlab (upstream). Will try again after next measurement: {e}")
                
            return

        except KeyboardInterrupt:
            logger.warning("Measurement interrupted, ramping down instruments")
            if rampdown_on_interrupt:
                rampdown_sweeps = [
                    Sweep(sweep.parameter, sweep.parameter(), 0.0, num=100, delay=1e-2)
                    for sweep in sweeps
                ]
                for sweep in self.buffered_dependents_tree:
                    for sw in self.buffered_dependents_tree[sweep]["sweeps"]:
                        rampdown_sweeps.append(
                            Sweep(sw.parameter, sw.parameter(), 0.0, num=100, delay=1e-2)
                        )
                sweeper(rampdown_sweeps)
                return 1
        finally:
            unregister_memory_store(self.id)


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
    )

    meas.run(sweeps, dependents, interrupt, rampdown_on_interrupt, verbose=verbose)
    if location_return:
        return meas.data
