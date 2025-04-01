import datetime
import json
import logging
import os
import socket
import subprocess
import sys
import uuid
from pathlib import Path, PurePosixPath
from typing import Callable, Sequence, Union
from checksumdir import dirhash

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

import numpy as np
import xarray as xr
from git import Repo, GitCommandError
from tabulate import tabulate
from tqdm import tqdm
from qcodes.parameters import Parameter

from qcutils.sweep import CircularSweep, Sweep, stepper, sweeper

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
        self.station = station
        logging.info(f"Measurement Location: {self.data}")

    def get_installed_packages(self):
        try:
            pip_output = subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"]
            ).decode()
        except subprocess.CalledProcessError as e:
            pip_output = ""

        try:
            uv_output = subprocess.check_output(["uv", "pip", "freeze"]).decode()
        except subprocess.CalledProcessError as e:
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
                "instrument_snapshot": str(dependent.instrument.snapshot()),
            },
        )

        data_array.data[:] = np.nan
        return data_array

    def _make_dataset(self, sweeps: Sequence[Sweep], dependents: list):
        """Create an xarray dataset for the measurement

        Args:
            sweeps (Sequence[Sweep]): List of sweep objects
            dependents (list): List of dependent QCoDeS parameters
        """
        code_path = Path(os.path.realpath(__file__)).parent
        code_archive = {}
        for file in os.listdir(code_path):
            try:
                with open(f"{code_path}/{file}", "r") as f:
                    code_archive[file] = f.read()
            except:
                pass

        try:
            cryostat_name = socket.gethostname().split(".")[0].split("-")[1]
        except:
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
            "Code Archive": str(code_archive),
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

        return xr.Dataset(data_vars=data_vars, coords=coords, attrs=meta)

    def _push_gitlab(self, dataset, data_hash):
        """Push the data to the gitlab repository

        Args:
            dataset (xarray.Dataset): xarray dataset
            data_hash (str): Hash of the data
        """
        git_ssh_identity_file = str(PurePosixPath(Path.home() / ".ssh" / "id_rsa"))
        git_ssh_cmd = f"ssh -i {git_ssh_identity_file}"

        if not os.path.exists(Path("~/.measurement-hashes/.git").expanduser()):
            logging.info("Cloning the measurement-hashes repository")
            Repo.clone_from(
                url="git@git.pgi.fz-juelich.de:squad-lab/hashes.git",
                to_path=self.git_repo,
                single_branch=True,
                branch=self.cryostat,
                env=dict(GIT_SSH_COMMAND=git_ssh_cmd),
            )
        repo = Repo(self.git_repo)

        if f"{self.cryostat}" not in repo.git.branch().split("* ")[1].split("\n"):
            logging.info(f"Creating and checking out to branch: {self.cryostat}")
            try:
                repo.git.checkout(b=f"{self.cryostat}")
            except GitCommandError:
                repo.git.branch(d=f"{self.cryostat}")
                repo.git.checkout(b=f"{self.cryostat}")
        else:
            logging.info(f"Checking out to branch:0 {self.cryostat}")
            repo.git.checkout(f"{self.cryostat}")

        hash_location = f"{self.git_repo}/{self.wafer_id}/{self.device_type}/{self.sample_name}/{self.experiment}"
        os.makedirs(hash_location, exist_ok=True)
        with open(
            f"{hash_location}/{self.id}",
            "w",
        ) as f:
            f.write(f"Hash: {data_hash}\n\n{json.dumps(dataset.attrs, indent=2)}")
        logging.info("Adding the measurement hash to the git repository")
        repo.git.add(all=True)
        repo.git.commit("-m", f"Add new measurement hash: {self.id}")
        repo.git.push("--set-upstream", "origin", f"{self.cryostat}")

    def run(
        self,
        sweeps: Union[Sweep, CircularSweep, Sequence[Sweep]],
        dependents: list,
        interrupt: Callable = lambda: False,
        rampdown_on_interrupt=False,
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

        sweep_dims = 0
        for sweep in sweeps:
            for param in sweep.parameter:
                independents.append(param)
                sweep_dims += 1

        # Get sweep metadata for pretty printed table
        sweep_metadata = []
        sweep_metadata_headers = [
            "Independent(s)",
            "Range",
            "Number of Points",
            "Delay (s)",
        ]

        for sweep in sweeps:
            sweep_metadata.append(
                [
                    ",".join([param.label for param in sweep.parameter]),
                    f"{min(sweep.values)}-{max(sweep.values)}",
                    len(sweep.values),
                    sweep.delay,
                ]
            )

        swm_list = [dict(zip(sweep_metadata_headers, swm)) for swm in sweep_metadata]
        meta = {
            "Instruments Snapshot": json.dumps(instruments_snapshot),
            "Parameters Snapshot": json.dumps(parameters_snapshot),
            "Sweeps": json.dumps({swm["Independent(s)"]: swm for swm in swm_list}),
            "Extra Metadata": json.dumps(self.extra_metadata),
        }
        self.arr = self._make_dataset(sweeps, dependents)
        self.arr.attrs.update(meta)
        self.arr.to_zarr(self.data, mode="a")

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

            logging.info(f"Starting the measurement with ID: {self.id}")
            global bar
            bar = tqdm(
                total=total_points,
                ascii="*ᗧⵔ●︎",
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
            )
            bar.close()

            data_hash = dirhash(self.data, "sha256", excluded_extensions=["pyc"])
            self._push_gitlab(dataset, data_hash)
            logging.info(f"Measurement completed and pushed with hash: {data_hash}")
            return

        except KeyboardInterrupt:
            logging.warning("Measurement interrupted, ramping donwn instruments")
            if rampdown_on_interrupt:
                rampdown_sweeps = [
                    Sweep(sweep.parameter, sweep.parameter(), 0.0, num=100, delay=1e-2)
                    for sweep in sweeps
                ]
                sweeper(rampdown_sweeps)
                return 1


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

    meas.run(sweeps, dependents, interrupt, rampdown_on_interrupt)
    if location_return:
        return meas.data
