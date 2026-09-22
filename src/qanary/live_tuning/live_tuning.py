from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Sequence

import numpy as np
import xarray as xr
import zarr
from qcodes.parameters import Parameter

from qanary.buffered.sweep import (
    _abort_instruments,
    _arm_instruments,
    _fetch_dependents_tree,
    _fetch_results,
)
from qanary.live_tuning.live_tuning_widgets import (
    LiveTuningWidgets,
)
from qanary.logger import get_logger
from qanary.measure import (
    Measurement,
    Station,
    _register_memory_store,
    _sanitize_for_json,
    _unregister_memory_store,
)
from qanary.sweep import Sweep, reset_disk_persist_cache

logger = get_logger(__name__)


# These are currently the DAC buffered nodes used by Qanary.
#
# Using module + class name here avoids importing hardware-specific modules
# when live_tuning itself is imported.
_SUPPORTED_DAC_NODES = {
    ("qcdrivers.buffered.basel.nodes", "NodeBaselDAC"),
    ("qcdrivers.buffered.qdevil.nodes", "NodeQDAC2"),
    ("qcdrivers.buffered.squad.nodes", "NodeDummySweeper"),  # used for testing
    ("__main__", "NodeDummySweeper"),  # used for testing
}


class ControlMailbox:
    """
    Thread-safe mailbox for parameters controlled interactively.

    Calling request() never accesses hardware. The acquisition thread calls
    apply_pending() between complete buffered frames and is therefore the only
    thread that writes to QCoDeS parameters.
    """

    def __init__(self, parameters: Sequence[Parameter]) -> None:
        self._parameters = {parameter.name: parameter for parameter in parameters}

        self._pending: dict[str, Any] = {}
        self._applied: dict[str, Any] = {}
        self._lock = Lock()

    @property
    def parameters(self) -> list[Parameter]:
        return list(self._parameters.values())

    def request(self, parameter: Parameter, value: Any) -> None:
        """
        Request a new parameter value.

        No hardware access occurs here. New values for the same parameter
        replace older pending values.
        """
        if parameter.name not in self._parameters:
            raise ValueError(
                f"{parameter.name!r} is not registered as a live tuning control."
            )

        if self._parameters[parameter.name] is not parameter:
            raise ValueError(
                f"A different parameter with name {parameter.name!r} "
                "was registered as the live tuning control."
            )

        with self._lock:
            # Coalescing:
            # 20 slider events become one hardware write with the newest value.
            self._pending[parameter.name] = value

    def apply_pending(self) -> dict[str, Any]:
        """
        Apply pending parameter changes.

        Must only be called from the acquisition thread.
        """
        with self._lock:
            pending = self._pending
            self._pending = {}

        for name, value in pending.items():
            parameter = self._parameters[name]

            logger.debug(
                "[live_tuning] Setting control %s = %s",
                name,
                value,
            )

            parameter(value)

            with self._lock:
                self._applied[name] = value

        return self.snapshot()

    def initialize(self) -> dict[str, Any]:
        """
        Read initial parameter values.

        Call this before acquisition starts, while no measurement thread owns
        the hardware yet.
        """
        values = {}

        for name, parameter in self._parameters.items():
            values[name] = parameter()

        with self._lock:
            self._applied = values

        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._applied)


class LiveTuning:
    """
    Continuously repeat one buffered 1D or 2D DAC measurement.

    Hardware access happens only on the acquisition thread.

    The IPython/Jupyter thread communicates with it through ControlMailbox.
    A separate publisher thread updates Qanary's MemoryStore at a fixed
    interval for Qimchi.

    On stop, the last completely fetched frame is persisted through the normal
    Qanary netCDF/hash path.
    """

    def __init__(
        self,
        *,
        controls: Sequence[Parameter] = (),
        control_ranges: Sequence[tuple[float, float]] = (),
        refresh_interval: float = 0.5,
        widget_step: float = 0.01,
        continuous_update: bool = True,
        show_widgets: bool = True,
        verbose: bool = False,
    ) -> None:

        if refresh_interval <= 0:
            raise ValueError("refresh_interval must be > 0.")

        self.refresh_interval = float(
            refresh_interval
        )
        self.verbose = verbose

        self._running = False

        # --------------------------------------------------
        # Controls
        # --------------------------------------------------

        self.controls = ControlMailbox(controls)

        # ---------------------------------------------------
        # State callbacks
        # ---------------------------------------------------

        self._state_callbacks = []

        # --------------------------------------------------
        # Run-specific state
        # --------------------------------------------------

        self._reset_run_state()

        # --------------------------------------------------
        # Widgets
        # --------------------------------------------------

        self.widgets = LiveTuningWidgets(
            tuning=self,
            controls=controls,
            ranges=control_ranges,
            step=widget_step,
            continuous_update=continuous_update,
            auto_display=show_widgets,
        )

    def set(
        self,
        parameter: Parameter,
        value: Any,
    ) -> None:
        """
        Request a new value for a live tuning control parameter.

        The parameter is not changed immediately in the notebook thread.
        Instead, the requested value is stored in the ControlMailbox and
        applied by the acquisition thread between buffered frames.

        Args:
            parameter: Registered live tuning control parameter.
            value: New target value.

        Raises:
            RuntimeError:
                If no live tuning measurement is currently running.

            ValueError:
                If the parameter was not registered as a control.
        """

        if not self._running:
            raise RuntimeError("No live tuning measurement is currently running.")

        self.controls.request(
            parameter,
            value,
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._running

    @property
    def frame_number(self) -> int:
        return self._frame_number

    @property
    def data_path(self) -> str:
        return self.measurement.data

    @property
    def dataset(self) -> xr.Dataset | None:
        """
        Return a copy of the latest complete buffered frame.
        """
        with self._dataset_lock:
            if self._latest_dataset is None:
                return None

            return self._latest_dataset.copy(deep=True)

    def _reset_run_state(self) -> None:
        """
        Reset all state belonging to one live-tuning run.
        """

        self._stop_event = Event()
        self._publisher_stop_event = Event()

        self._dataset_lock = Lock()
        self._publish_lock = Lock()

        self._acquisition_thread = None
        self._publisher_thread = None

        self._template = None
        self._latest_dataset = None

        self._frame_number = 0
        self._dataset_version = 0
        self._published_version = -1

        self._worker_error = None
        self._publisher_error = None

        self._snapshot_before = None
        self._snapshot_after = None

    def _format_snapshot_value(self, value: Any, max_length: int = 50) -> str:
        """Format a value for the console table."""
        if isinstance(value, np.ndarray):
            value = value.tolist()

        value_str = str(value)

        if len(value_str) > max_length:
            value_str = value_str[:max_length] + "..."

        return value_str

    def _parameter_changes(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> dict[str, Any]:

        changes = {}

        before_parameters = before["parameters"]
        after_parameters = after["parameters"]

        for name in before_parameters.keys() | after_parameters.keys():
            before_entry = before_parameters.get(name)
            after_entry = after_parameters.get(name)

            if before_entry is None or after_entry is None:
                continue

            old = before_entry.get("value")
            new = after_entry.get("value")

            if old != new:
                changes[name] = {
                    "before": old,
                    "after": new,
                    "unit": after_entry.get("unit", ""),
                    "label": after_entry.get("label", name),
                }

        return changes

    def _sweep_snapshot(self) -> list[dict[str, Any]]:
        """
        Return metadata for the buffered DAC sweeps.

        LiveTuning only supports the root buffered DAC sweeps, so this is
        intentionally simpler than Measurement.run().
        """
        result = []

        for sweep in self.buffered_sweep["sweeps"]:
            result.append(
                {
                    "Independent(s)": ", ".join(
                        parameter.label for parameter in sweep.parameter
                    ),
                    "Parameter Name(s)": [
                        parameter.name for parameter in sweep.parameter
                    ],
                    "Range": [
                        _sanitize_for_json(sweep.values[0]),
                        _sanitize_for_json(sweep.values[-1]),
                    ],
                    "Number of Points": len(sweep.values),
                    "Delay (s)": sweep.delay,
                    "Start Delay (s)": getattr(
                        sweep,
                        "start_delay",
                        None,
                    ),
                    "Buffered": "true",
                }
            )

        return result

    def _take_snapshot(self, phase: str) -> dict[str, Any]:
        """
        Take a complete station snapshot.

        Must only be called while the acquisition thread is not accessing
        hardware:
            - before acquisition starts
            - after acquisition has stopped
        """
        timestamp = datetime.now().isoformat()

        station = self.measurement.station

        # --------------------------------------------------------------
        # Instrument snapshots
        # --------------------------------------------------------------

        instruments_snapshot = {}

        for instrument in station.instruments:
            try:
                snapshot = instrument.snapshot()

            except Exception as exc:
                logger.exception(
                    "Failed taking %s snapshot of instrument %s",
                    phase,
                    instrument.name,
                )

                snapshot = {
                    "snapshot_error": str(exc),
                }

            instruments_snapshot[instrument.name] = _sanitize_for_json(snapshot)

        # --------------------------------------------------------------
        # Registered Qanary parameters
        # --------------------------------------------------------------

        parameters_snapshot = {}
        parameter_table = []

        for parameter in station.parameters:
            try:
                value = parameter()

            except Exception as exc:
                logger.exception(
                    "Failed reading parameter %s for %s snapshot",
                    parameter.name,
                    phase,
                )

                value = None
                error = str(exc)

            else:
                error = None

            value = _sanitize_for_json(value)

            entry = {
                "value": value,
                "unit": parameter.unit,
                "label": parameter.label,
                "param_type": getattr(
                    parameter,
                    "param_type",
                    None,
                ),
            }

            if error is not None:
                entry["error"] = error

            parameters_snapshot[parameter.name] = entry

            parameter_table.append(
                [
                    parameter.name,
                    parameter.label,
                    self._format_snapshot_value(value),
                    parameter.unit,
                ]
            )

        # --------------------------------------------------------------
        # Sweep definition
        # --------------------------------------------------------------

        sweeps_snapshot = self._sweep_snapshot()

        # --------------------------------------------------------------
        # Console output
        # --------------------------------------------------------------

        logger.info(
            "Registered Parameters (%s):",
            phase.capitalize(),
        )

        print("\n")

        self.measurement._print_table(
            snapshot_table=parameter_table,
            headers=[
                "Name",
                "Label",
                "Value",
                "Unit",
            ],
        )

        print("\n")

        sweep_table = []

        for sweep in sweeps_snapshot:
            start, stop = sweep["Range"]

            try:
                range_string = f"{float(start):.3e} to {float(stop):.3e}"
            except (TypeError, ValueError):
                range_string = f"{start} to {stop}"

            sweep_table.append(
                [
                    sweep["Independent(s)"],
                    range_string,
                    sweep["Number of Points"],
                    sweep["Delay (s)"],
                ]
            )

        logger.info(
            "Sweeps Summary (%s):",
            phase.capitalize(),
        )

        print("\n")

        self.measurement._print_table(
            snapshot_table=sweep_table,
            headers=[
                "Independent(s)",
                "Range",
                "Number of Points",
                "Delay (s)",
            ],
        )

        print("\n")

        return {
            "timestamp": timestamp,
            "parameters": parameters_snapshot,
            "instruments": instruments_snapshot,
            "sweeps": sweeps_snapshot,
        }

    def start(
        self,
        sweeps,
        dependents=(),
        *,
        wafer_id: str,
        device_type: str,
        sample_name: str,
        station: Station,
        experiment_name: str,
        data_location: str,
        metadata: dict | None = None,
        fridge_name: str = "",
        save_interval: float = 0.1,
        nc_snapshot_during_run: bool = False,
        git_repo: str = "~/.measurement-hashes",
        no_hashing: bool = False,
        location_return: bool = False,
        verbose: bool | None = None,
    ) -> LiveTuning:
        """
        Prepare publication and start acquisition in the background.

        Returns immediately so IPython remains interactive.
        """
        if self._running:
            raise RuntimeError("Live tuning session is already running.")

        self._reset_run_state()

        if metadata is None:
            metadata = {}

        if verbose is None:
            verbose = self.verbose

        self.no_hashing = no_hashing
        self.verbose = verbose
        self.station = station

        if not isinstance(sweeps, (list, tuple)):
            sweeps = [sweeps]

        if len(sweeps) != 1:
            raise ValueError(
                "Live tuning currently accepts exactly one buffered DAC sweep tree."
            )

        buffered_sweep = sweeps[0]

        if not isinstance(buffered_sweep, dict):
            raise TypeError("Live tuning requires a buffered sweep tree.")

        self.buffered_sweep = buffered_sweep

        self._validate_buffered_sweep()

        self.measurement = Measurement(
            wafer_id=wafer_id,
            device_type=device_type,
            sample_name=sample_name,
            experiment_name=experiment_name,
            station=self.station,
            data_location=data_location,
            metadata=metadata,
            fridge_name=fridge_name,
            save_interval=save_interval,
            nc_snapshot_during_run=nc_snapshot_during_run,
            git_repo=git_repo,
        )

        self._prepare()

        self._running = True

        ### start two threads ###
        self._publisher_thread = Thread(
            target=self._publisher_loop,
            name=f"qanary-live-publisher-{self.measurement.id}",
            daemon=True,
        )

        self._acquisition_thread = Thread(
            target=self._acquisition_loop,
            name=f"qanary-live-acquisition-{self.measurement.id}",
            daemon=True,
        )
        #########################

        self._notify_state()

        # Publisher first so Qimchi can already see the empty/preallocated
        # dataset while the first hardware frame is running.
        self._publisher_thread.start()
        self._acquisition_thread.start()

        logger.info(
            "Live tuning started with measurement ID %s",
            self.measurement.id,
        )

        return self

    def add_state_callback(self, callback) -> None:
        """
        Register a callback that is called whenever the running state changes.

        callback(running: bool)
        """
        self._state_callbacks.append(callback)

    def _notify_state(self) -> None:
        for callback in self._state_callbacks:
            try:
                callback(self._running)
            except Exception:
                logger.exception("Live tuning state callback failed.")

    def stop(self) -> xr.Dataset:
        """
        Stop after the currently running buffered frame and finalize the data.

        The buffered sweep is treated atomically: stop() does not modify DAC
        state from the notebook thread. Therefore a currently executing
        hardware sweep is allowed to finish, its results are fetched, and that
        complete frame becomes the final dataset.
        """
        if not self._running:
            raise RuntimeError("Live tuning session has not been started.")

        # Cooperative stop. The acquisition thread remains the sole hardware
        # owner and notices this after the current buffered frame.
        self._stop_event.set()

        if self._acquisition_thread is not None:
            self._acquisition_thread.join()

        self._running = False
        self._notify_state()

        # No new datasets can appear after acquisition has stopped.
        # Publish the newest frame once more.
        self._publish_latest(force=True)

        self._publisher_stop_event.set()

        if self._publisher_thread is not None:
            self._publisher_thread.join()

        final_dataset = self.dataset

        if final_dataset is None:
            raise RuntimeError("No dataset exists to finalize.")

        # snapshot after the live measurement
        self._snapshot_after = self._take_snapshot(phase="after")

        final_dataset.attrs["Measurement Mode"] = "live_tuning"
        final_dataset.attrs["Live Tuning State"] = "finished"
        final_dataset.attrs["Live Tuning Frames"] = self._frame_number

        final_dataset.attrs["Live Tuning End"] = self._snapshot_after["timestamp"]

        final_dataset.attrs["Snapshot After"] = json.dumps(
            _sanitize_for_json(self._snapshot_after)
        )

        final_dataset.attrs["Parameters Snapshot After"] = json.dumps(
            _sanitize_for_json(self._snapshot_after["parameters"])
        )

        final_dataset.attrs["Instruments Snapshot After"] = json.dumps(
            _sanitize_for_json(self._snapshot_after["instruments"])
        )

        # track changes
        changes = self._parameter_changes(
            self._snapshot_before,
            self._snapshot_after,
        )

        final_dataset.attrs["Parameter Changes"] = json.dumps(
            _sanitize_for_json(changes)
        )

        ###

        final_dataset.attrs["Live Tuning"] = "true"
        final_dataset.attrs["Live Tuning State"] = "finished"
        final_dataset.attrs["Live Tuning Frames"] = self._frame_number
        final_dataset.attrs["Live Tuning End"] = datetime.now().isoformat()
        final_dataset.attrs["Live Tuning Controls"] = json.dumps(
            _sanitize_for_json(self.controls.snapshot())
        )

        self.measurement.arr = final_dataset

        try:
            exported = self.measurement._finalize_disk_artifacts(
                dataset=final_dataset,
                verbose=self.verbose,
            )

            if exported is None:
                raise RuntimeError(
                    "Live tuning stopped, but final netCDF export failed."
                )

            if not self.no_hashing:
                data_hash = self.measurement._compute_data_hash()

                try:
                    self.measurement._push_gitlab(exported, data_hash)
                except Exception:
                    # Match normal Measurement behaviour: the data is valid
                    # even when pushing the hash repository fails.
                    logger.exception(
                        "Final data was saved and hashed, but the hash "
                        "could not be pushed upstream."
                    )
                else:
                    logger.info(
                        "Live tuning completed with hash: %s",
                        data_hash,
                    )

        finally:
            _unregister_memory_store(self.measurement.id)

        if self._worker_error is not None:
            raise RuntimeError(
                "Live tuning acquisition failed. The last complete frame was finalized."
            ) from self._worker_error

        # return exported
        return None

    close = stop

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    def _validate_buffered_sweep(self) -> None:
        if not isinstance(self.buffered_sweep, dict):
            raise TypeError(
                "LiveTuning only accepts one buffered sweep tree as a dict."
            )

        root = self.buffered_sweep

        if "instrument" not in root:
            raise ValueError("Buffered live tuning sweep requires a root 'instrument'.")

        root_instrument = root["instrument"]

        node_type = (
            root_instrument.__class__.__module__,
            root_instrument.__class__.__name__,
        )

        if node_type not in _SUPPORTED_DAC_NODES:
            supported = ", ".join(
                sorted(class_name for _, class_name in _SUPPORTED_DAC_NODES)
            )
            raise TypeError(
                "Live tuning only supports buffered DAC root nodes. "
                f"Received {node_type[1]!r}. Supported: {supported}."
            )

        sweeps = root.get("sweeps")

        if not isinstance(sweeps, (list, tuple)):
            raise TypeError("Buffered DAC root must contain a 'sweeps' list.")

        if len(sweeps) not in (1, 2):
            raise ValueError(
                "Live tuning supports exactly one or two buffered DAC sweeps."
            )

        swept_parameters = []

        for index, sweep in enumerate(sweeps):
            if not isinstance(sweep, Sweep):
                raise TypeError(
                    "Live tuning only supports qanary.sweep.Sweep "
                    f"for buffered DAC axes. Axis {index} is "
                    f"{type(sweep).__name__}."
                )

            if len(sweep.parameter) != 1:
                raise ValueError(
                    "Each live tuning sweep axis must control exactly "
                    "one DAC parameter."
                )

            swept_parameters.append(sweep.parameter[0])

        if len({id(parameter) for parameter in swept_parameters}) != len(
            swept_parameters
        ):
            raise ValueError(
                "The same DAC parameter cannot be used for both sweep axes."
            )

        control_ids = {id(parameter) for parameter in self.controls.parameters}

        for parameter in swept_parameters:
            if id(parameter) in control_ids:
                raise ValueError(
                    f"{parameter.name!r} cannot simultaneously be a "
                    "buffered sweep axis and a live tuning control."
                )

        # Live tuning intentionally supports only ONE DAC sweep block.
        # Children may acquire dependents, but may not introduce additional
        # nested buffered sweeps.
        def validate_children(node: dict[str, Any]) -> None:
            children = node.get("nodes", []) or []

            if not isinstance(children, list):
                raise TypeError("'nodes' must be a list.")

            for child in children:
                if not isinstance(child, dict):
                    raise TypeError("Each buffered-tree child must be a dict.")

                if child.get("sweeps"):
                    raise ValueError(
                        "Live tuning does not support nested buffered sweeps. "
                        "Only the root DAC may contain sweeps."
                    )

                validate_children(child)

        validate_children(root)

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def _prepare(self) -> None:
        """
        Build the xarray structure and register it with Qimchi.
        """
        # Discover buffered dependents and their sweep dimensions using
        # Qanary's existing buffered-tree machinery.
        dependent_state: dict[str, Any] = {}

        _fetch_dependents_tree(
            self.buffered_sweep,
            state=dependent_state,
        )

        dependent_tree = dependent_state.get("dependent_tree", {})

        if not dependent_tree:
            raise ValueError(
                "Live tuning needs at least one buffered dependent to plot."
            )

        self.measurement.buffered_dependents_tree = dependent_tree

        # _make_dataset already understands buffered dependents if
        # buffered_dependents_tree has been populated.
        template = self.measurement._make_dataset(
            [self.buffered_sweep],
            dependents=[],
        )

        # snapshot handling
        self._snapshot_before = self._take_snapshot(phase="before")

        template.attrs["Measurement Mode"] = "live_tuning"
        template.attrs["Live Tuning State"] = "running"
        template.attrs["Live Tuning Frames"] = 0
        template.attrs["Live Tuning Start"] = self._snapshot_before["timestamp"]

        template.attrs["Snapshot Before"] = json.dumps(
            _sanitize_for_json(self._snapshot_before)
        )

        template.attrs["Parameters Snapshot Before"] = json.dumps(
            _sanitize_for_json(self._snapshot_before["parameters"])
        )

        template.attrs["Instruments Snapshot Before"] = json.dumps(
            _sanitize_for_json(self._snapshot_before["instruments"])
        )

        template.attrs["Sweeps"] = json.dumps(
            _sanitize_for_json(self._snapshot_before["sweeps"])
        )
        ###

        self.controls.initialize()

        template.attrs.update(self._build_metadata())
        template.attrs["Live Tuning"] = "true"
        template.attrs["Live Tuning State"] = "running"
        template.attrs["Live Tuning Frames"] = 0
        template.attrs["Live Tuning Start"] = datetime.now().isoformat()
        template.attrs["Live Tuning Controls"] = json.dumps(
            _sanitize_for_json(self.controls.snapshot())
        )

        self._template = template

        with self._dataset_lock:
            self._latest_dataset = template.copy(deep=True)
            self._dataset_version = 0

        self.measurement.arr = self._latest_dataset

        # Reuse exactly the same MemoryStore / temporary disk store mechanism
        # as normal Qanary measurements.
        self.measurement.memory_store = zarr.MemoryStore()
        self.measurement.disk_store = zarr.DirectoryStore(self.measurement.live_data)

        template.to_zarr(
            store=self.measurement.memory_store,
            mode="w",
        )

        zarr.copy_store(
            self.measurement.memory_store,
            self.measurement.disk_store,
            if_exists="replace",
        )

        reset_disk_persist_cache(self.measurement.disk_store)

        _register_memory_store(
            self.measurement.id,
            self.measurement.memory_store,
            disk_path=str(Path(self.measurement.live_data).resolve()),
        )

    def _build_metadata(self) -> dict[str, Any]:
        """
        Produce metadata similar to Measurement.run().
        """
        station = self.measurement.station

        instrument_snapshots = {
            instrument.name: instrument.snapshot() for instrument in station.instruments
        }

        parameter_snapshots = {}

        for parameter in station.parameters:
            value = parameter()

            if isinstance(value, np.ndarray):
                value = value.tolist()

            parameter_snapshots[parameter.name] = {
                "value": value,
                "unit": parameter.unit,
                "label": parameter.label,
            }

        sweeps = self.buffered_sweep["sweeps"]

        sweep_metadata = {}

        for sweep in sweeps:
            parameter = sweep.parameter[0]

            sweep_metadata[parameter.name] = {
                "Independent(s)": parameter.label,
                "Range": [
                    float(sweep.values[0]),
                    float(sweep.values[-1]),
                ],
                "Number of Points": len(sweep.values),
                "Delay (s)": sweep.delay,
                "Buffered": True,
            }

        return {
            "Instruments Snapshot": json.dumps(
                _sanitize_for_json(instrument_snapshots)
            ),
            "Parameters Snapshot": json.dumps(_sanitize_for_json(parameter_snapshots)),
            "Sweeps": json.dumps(_sanitize_for_json(sweep_metadata)),
            "Extra Metadata": json.dumps(
                _sanitize_for_json(self.measurement.extra_metadata)
            ),
        }

    # ------------------------------------------------------------------
    # acquisition
    # ------------------------------------------------------------------

    def _acquisition_loop(self) -> None:
        """
        Continuously acquire complete buffered frames.
        """
        try:
            while not self._stop_event.is_set():
                # Apply slider changes ONLY here.
                #
                # The notebook thread never touches hardware.
                control_values = self.controls.apply_pending()

                frame = self._acquire_frame(control_values)

                self._commit_frame(frame)

        except BaseException as exc:
            self._worker_error = exc
            self._stop_event.set()

            logger.exception("Live tuning acquisition failed")

            # abort() also stays on the hardware-owning acquisition thread.
            try:
                _abort_instruments(self.buffered_sweep)
            except Exception:
                logger.exception(
                    "Failed aborting buffered instruments after live tuning error."
                )

    def _acquire_frame(
        self,
        control_values: dict[str, Any],
    ) -> xr.Dataset:
        """
        Execute one complete hardware-buffered DAC frame.
        """
        if self._template is None:
            raise RuntimeError("Live tuning has not been prepared.")

        # Fresh frame: only complete frames are ever committed.
        frame = self._template.copy(deep=True)

        toplevel = _arm_instruments(self.buffered_sweep)

        # This call blocks ONLY the background acquisition thread.
        toplevel.run_sweep()

        results_state: dict[str, Any] = {}

        _fetch_results(
            self.buffered_sweep,
            state=results_state,
        )

        results_tree = results_state.get("results_tree", {})

        if not results_tree:
            raise RuntimeError(
                "Buffered sweep completed but returned no dependent data."
            )

        for dependent, entry in results_tree.items():
            variable_name = dependent.name

            if variable_name not in frame.data_vars:
                raise RuntimeError(f"Fetched unknown dependent {variable_name!r}.")

            result = np.asarray(entry["result"])

            expected_shape = frame[variable_name].shape

            if result.shape != expected_shape:
                raise RuntimeError(
                    f"Buffered dependent {variable_name!r} returned "
                    f"shape {result.shape}, expected {expected_shape}."
                )

            frame[variable_name].data[...] = result

        next_frame = self._frame_number + 1

        frame.attrs["Live Tuning"] = "true"
        frame.attrs["Live Tuning State"] = "running"
        frame.attrs["Live Tuning Frame"] = next_frame
        frame.attrs["Live Tuning Frames"] = next_frame
        frame.attrs["Live Tuning Frame Timestamp"] = datetime.now().isoformat()
        frame.attrs["Live Tuning Controls"] = json.dumps(
            _sanitize_for_json(control_values)
        )

        return frame

    def _commit_frame(self, frame: xr.Dataset) -> None:
        """
        Atomically replace the Python-side reference to the latest frame.

        The frame is never modified after this point.
        """
        with self._dataset_lock:
            self._latest_dataset = frame
            self._frame_number += 1
            self._dataset_version += 1

            self.measurement.arr = frame

        logger.debug(
            "[live_tuning] Completed frame %d",
            self._frame_number,
        )

    # ------------------------------------------------------------------
    # Qimchi publication
    # ------------------------------------------------------------------

    def _publisher_loop(self) -> None:
        """
        Publish the newest complete frame at a fixed maximum refresh rate.

        The acquisition speed and plot refresh speed are independent.
        """
        # Initial empty/preallocated dataset.
        self._publish_latest(force=True)

        while not self._publisher_stop_event.wait(self.refresh_interval):
            try:
                self._publish_latest()
            except Exception:
                logger.exception("Failed publishing live tuning dataset.")

    def _publish_latest(self, *, force: bool = False) -> None:
        with self._publish_lock:
            with self._dataset_lock:
                dataset = self._latest_dataset
                version = self._dataset_version

            if dataset is None:
                return

            if not force and version == self._published_version:
                return

            memory_store = self.measurement.memory_store
            disk_store = self.measurement.disk_store

            if memory_store is None:
                return

            # Dataset structure is fixed for the entire tuning session.
            # Therefore only data variables + root attrs need updating.
            memory_root = zarr.open_group(
                store=memory_store,
                mode="r+",
            )

            for name, variable in dataset.data_vars.items():
                memory_root[name][...] = variable.values

            memory_root.attrs.update(dict(dataset.attrs))

            # Temporary disk copy for crash recovery.
            if disk_store is not None:
                disk_root = zarr.open_group(
                    store=disk_store,
                    mode="r+",
                )

                for name, variable in dataset.data_vars.items():
                    disk_root[name][...] = variable.values

                disk_root.attrs.update(dict(dataset.attrs))

            self._published_version = version

            if self.verbose:
                logger.info(
                    "[live_tuning] Published frame %d",
                    self._frame_number,
                )
