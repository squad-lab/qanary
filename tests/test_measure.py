"""
Tests for ``qcutils.measure`` -- the station, the measurement and the run.

``Measurement.run`` is where everything else meets: it snapshots the
instruments, allocates the dataset, seeds the in-memory and on-disk Zarr
stores, registers the run for live plotting, drives ``_stepper``, exports
netCDF and pushes a hash. Most of it is bookkeeping, and bookkeeping is
exactly what silently produces a dataset nobody can interpret six months
later, so the metadata assertions here are as load-bearing as the numeric
ones.

The tests run real measurements against ``DummyInstrument``. Only three things
are faked: the package freeze (a subprocess call worth seconds per run), the
WebSocket server (Qimchi Connect's concern) and, where a test says so, the
GitLab hash push.

"""

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
import zarr
from git import Repo
from qcodes.parameters import Parameter
from qimchi_connect import close_live_measurement, get_live_registration
from qimchi_connect import registry as live_db
from qimchi_connect import server as live_server

from qcutils import measure
from qcutils.measure import (
    Measurement,
    Station,
    _find_available_port,
    _register_memory_store,
    _sanitize_for_json,
    _unregister_memory_store,
    run,
)
from qcutils.sweep import Sweep


class TestSanitizeForJson:
    """
    Instrument snapshots contain whatever a driver cached, and a single
    unserialisable object would otherwise lose the entire metadata blob.

    """

    def test_numpy_arrays_become_lists(self):
        assert _sanitize_for_json(np.arange(3)) == [0, 1, 2]

    def test_numpy_scalars_become_python_scalars(self):
        result = _sanitize_for_json(np.float64(1.5))

        assert result == 1.5
        assert isinstance(result, float)

    def test_containers_are_walked_recursively(self):
        sanitized = _sanitize_for_json(
            {"a": [np.int64(1), (np.float32(2.0),)], "b": {"c": np.arange(2)}}
        )

        assert sanitized == {"a": [1, [2.0]], "b": {"c": [0, 1]}}
        assert json.dumps(sanitized)

    def test_sets_become_lists(self):
        assert sorted(_sanitize_for_json({1, 2})) == [1, 2]

    def test_unserialisable_objects_become_their_repr(self):
        """
        The driver context objects this exists for -- stringifying keeps the
        snapshot readable instead of dropping the parameter entirely.

        """

        class TriggerContext:
            def __repr__(self):
                return "<QDac2Trigger_Context 3>"

        assert _sanitize_for_json(TriggerContext()) == "<QDac2Trigger_Context 3>"

    def test_ordinary_values_pass_through_unchanged(self):
        payload = {"name": "dmm", "count": 3, "ok": True, "missing": None}

        assert _sanitize_for_json(payload) == payload

    def test_the_result_is_always_serialisable(self):
        snapshot = {"param": {"value": np.arange(2), "context": object()}}

        assert json.dumps(_sanitize_for_json(snapshot))


class TestFindAvailablePort:
    """Picking the WebSocket port for live plotting."""

    def test_returns_a_port_that_can_be_bound(self):
        port = _find_available_port(8765)

        assert 8765 <= port < 8865

    def test_skips_a_port_already_in_use(self):
        """
        Two measurements in one session must not be told to serve on the same
        port. Only "not the busy one" is asserted -- which of the following
        ports is free is the machine's business.

        """
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
            taken.bind(("localhost", 0))
            busy = taken.getsockname()[1]

            found = _find_available_port(busy)

        assert busy < found <= busy + 100

    def test_falls_back_to_the_starting_port_when_none_are_free(self, monkeypatch):
        """
        Returning the start port rather than raising means the caller reports
        a failed server start, which is degraded live plotting rather than a
        failed measurement.

        """

        class AlwaysBusy:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def bind(self, address):
                raise OSError("address in use")

        monkeypatch.setattr("qcutils.measure.socket.socket", lambda *a: AlwaysBusy())

        assert _find_available_port(8765, max_attempts=3) == 8765


class TestLiveStoreRegistry:
    """Publishing a memory store as a live measurement."""

    def test_registering_publishes_the_store_and_its_disk_path(
        self, live_registry, offline_live_server, quiet_registry_maintenance, tmp_path
    ):
        store = zarr.MemoryStore()
        disk_path = str(tmp_path / "1-abc.zarr")

        port = _register_memory_store("1-abc", store, disk_path=disk_path)

        assert port == offline_live_server
        assert live_server._provider_entry("1-abc") is not None
        assert get_live_registration("1-abc").disk_path == disk_path

    def test_registering_writes_a_row_to_the_live_database(
        self, live_registry, offline_live_server, quiet_registry_maintenance, tmp_path
    ):
        """
        Qimchi discovers running measurements through this row, so a
        registration that never reaches the database is invisible to it.

        """
        _register_memory_store(
            "1-db", zarr.MemoryStore(), disk_path=str(tmp_path / "1-db.zarr")
        )

        record = live_db.get_measurement("1-db")
        assert record is not None
        assert record.ws_port == offline_live_server

    def test_a_registration_without_a_disk_path_is_still_recorded(
        self, live_registry, offline_live_server, quiet_registry_maintenance
    ):
        """
        A measurement that has not been written to disk yet is still live, so
        Qimchi should be able to find it; the row simply carries no path.

        """
        _register_memory_store("1-memory-only", zarr.MemoryStore())

        record = live_db.get_measurement("1-memory-only")
        assert record is not None
        assert record.fpath == ""
        assert live_server._provider_entry("1-memory-only") is not None

    def test_a_failing_maintenance_pass_does_not_block_registration(
        self, live_registry, offline_live_server, caplog
    ):
        """
        Maintenance is housekeeping over old rows; a locked database must not
        stop a measurement from starting.

        """
        with patch.object(
            live_db, "maintain_registry", side_effect=RuntimeError("locked")
        ):
            _register_memory_store("1-unmaintained", zarr.MemoryStore())

        # Registry maintenance and its warning belong to Qimchi Connect.
        assert "Could not maintain the live registry" in caplog.text
        assert live_server._provider_entry("1-unmaintained") is not None

    def test_maintenance_findings_are_logged(
        self, live_registry, offline_live_server, caplog
    ):
        with patch.object(live_db, "maintain_registry") as maintain:
            maintain.return_value = SimpleNamespace(
                stale_measurement_ids=("1-stale",), deleted_count=2
            )
            _register_memory_store("1-fresh", zarr.MemoryStore())

        assert "marked 1 stale and deleted 2" in caplog.text

    def test_a_started_server_reports_its_port(
        self, live_registry, quiet_registry_maintenance
    ):
        """The path taken by the first measurement in a session."""
        with (
            patch.object(live_server, "is_server_running", return_value=False),
            patch.object(live_server, "start_live_server", return_value=True),
            patch.object(live_server, "get_server_port", return_value=8765),
            patch.object(live_server, "get_server_host", return_value="localhost"),
            patch.object(measure, "_find_available_port", return_value=8765),
        ):
            port = _register_memory_store("1-starting", zarr.MemoryStore())

        assert port == 8765
        assert live_db.get_measurement("1-starting").ws_url == "ws://localhost:8765"
        close_live_measurement("1-starting")

    def test_a_server_that_will_not_start_leaves_the_port_at_zero(
        self, live_registry, quiet_registry_maintenance, caplog, tmp_path
    ):
        """
        Live plotting is optional, so failure to start the server must not stop
        the measurement. Qimchi Connect removes the unused provider and writes
        no discovery row; the QCUtils wrapper reports the failure as port zero.

        """
        with (
            patch.object(live_server, "is_server_running", return_value=False),
            patch.object(live_server, "start_live_server", return_value=False),
        ):
            port = _register_memory_store(
                "1-serverless",
                zarr.MemoryStore(),
                disk_path=str(tmp_path / "x.zarr"),
            )

        assert port == 0
        assert "Failed publishing live measurement 1-serverless" in caplog.text
        assert live_db.get_measurement("1-serverless") is None
        assert live_server._provider_entry("1-serverless") is None

    def test_a_failing_database_registration_does_not_stop_the_run(
        self,
        live_registry,
        offline_live_server,
        quiet_registry_maintenance,
        tmp_path,
        caplog,
    ):
        with patch.object(
            live_db, "register_measurement", side_effect=RuntimeError("locked")
        ):
            port = _register_memory_store(
                "1-unrecorded",
                zarr.MemoryStore(),
                disk_path=str(tmp_path / "x.zarr"),
            )

        # The run continues without live plotting rather than raising. Without
        # a discovery row, keeping the provider would not make it discoverable.
        assert port == 0
        assert "Failed publishing live measurement 1-unrecorded" in caplog.text
        assert live_server._provider_entry("1-unrecorded") is None

    def test_publishing_several_measurements_advertises_all_of_them(
        self, live_registry, offline_live_server, quiet_registry_maintenance
    ):
        _register_memory_store("1-a", zarr.MemoryStore())
        _register_memory_store("1-b", zarr.MemoryStore())

        assert set(live_server._measurement_ids()) == {"1-a", "1-b"}

    def test_unregistering_removes_the_store_and_ends_the_database_row(
        self, live_registry, offline_live_server, quiet_registry_maintenance, tmp_path
    ):
        _register_memory_store(
            "1-ending", zarr.MemoryStore(), disk_path=str(tmp_path / "x.zarr")
        )

        _unregister_memory_store("1-ending")

        assert live_server._provider_entry("1-ending") is None
        assert live_db.get_measurement("1-ending").ended_at is not None

    def test_unregistering_an_unknown_measurement_is_harmless(
        self, live_registry, offline_live_server
    ):
        _unregister_memory_store("never-registered")

    def test_a_failing_database_does_not_break_unregistration(
        self, offline_live_server, caplog
    ):
        """
        Unregistration runs in a ``finally``; raising here would mask whatever
        actually ended the measurement.

        """
        with patch.object(
            measure, "close_live_measurement", side_effect=RuntimeError("locked")
        ):
            _unregister_memory_store("1-x")

        assert "Failed closing live measurement 1-x" in caplog.text


class TestStation:
    """The station is the list of parameters that gets snapshotted."""

    def test_a_single_parameter_is_added_with_its_alias(self, gates):
        built = Station("s")

        added = built.add_parameter("plunger", "Plunger 1", gates.x)

        assert added.name == "plunger"
        assert added.label == "Plunger 1"
        assert built.parameters == [added]

    def test_a_sequence_of_parameters_becomes_a_multi_channel_parameter(self, gates):
        built = Station("s")

        added = built.add_parameter("both", "Both gates", [gates.x, gates.y])

        added(0.4)
        assert gates.x() == pytest.approx(0.4)
        assert gates.y() == pytest.approx(0.4)

    def test_the_parameter_type_is_recorded(self, gates):
        built = Station("s")

        added = built.add_parameter("ohmic", "Ohmic", gates.x, param_type="ohmic")

        assert added.param_type == "ohmic"

    def test_a_parameter_can_be_removed(self, gates):
        built = Station("s")
        added = built.add_parameter("plunger", "Plunger", gates.x)

        built.remove_parameter(added)

        assert built.parameters == []

    def test_removing_an_unknown_parameter_is_an_error(self, gates):
        built = Station("s")

        with pytest.raises(ValueError, match="not found in station"):
            built.remove_parameter(gates.x)

    def test_the_duplicate_guard_rejects_a_repeated_name(self, gates):
        """
        ``add_parameter`` used to compare the freshly built wrapper against the
        list. A new wrapper is built on every call and Parameter compares by
        identity, so the guard never fired and the same gate could be
        registered -- and snapshotted -- twice. It now compares by name.

        """
        built = Station("s")
        built.add_parameter("plunger", "Plunger", gates.x)

        with pytest.raises(ValueError, match="already exists in station"):
            built.add_parameter("plunger", "Plunger", gates.x)

        assert len(built.parameters) == 1

    def test_distinct_names_are_still_accepted(self, gates):
        """The guard keys on the registered name, not on the qcodes parameter."""
        built = Station("s")

        built.add_parameter("plunger", "Plunger", gates.x)
        built.add_parameter("barrier", "Barrier", gates.x)

        assert len(built.parameters) == 2


class TestMeasurementSetup:
    """Where a measurement decides to write, and under what identity."""

    def test_creates_its_directory_tree_from_the_sample_hierarchy(
        self, measurement, tmp_path
    ):
        expected = tmp_path / "data" / "W1" / "hall-bar" / "S1" / "pinchoff"

        assert Path(measurement.datalogging) == expected
        assert expected.is_dir()

    def test_the_first_measurement_is_numbered_one(self, measurement):
        assert measurement.id.startswith("1-")

    def test_the_identifier_carries_a_uuid(self, measurement):
        """
        The counter is per-directory, so the uuid is what makes the identifier
        unique across samples and machines.

        """
        assert len(measurement.id.split("-", 1)[1].split("-")) == 5

    def test_the_number_continues_from_the_files_already_there(
        self, tmp_path, station, live_registry, offline_live_server
    ):
        folder = tmp_path / "data" / "W1" / "hall-bar" / "S1" / "pinchoff"
        folder.mkdir(parents=True)
        (folder / "1-aaa.nc").touch()
        (folder / "7-bbb.zarr").mkdir()
        (folder / "notes.md").touch()

        with patch.object(Measurement, "get_installed_packages", return_value=""):
            following = Measurement(
                "W1",
                "hall-bar",
                "S1",
                "pinchoff",
                station,
                str(tmp_path / "data"),
                {},
            )

        assert following.id.startswith("8-")

    def test_the_netcdf_and_live_zarr_paths_share_a_stem(self, measurement):
        assert measurement.data.endswith(".nc")
        assert measurement.live_data.endswith(".zarr")
        assert Path(measurement.data).stem == Path(measurement.live_data).stem

    def test_a_fridge_name_gives_the_hash_repository_its_own_clone(self, measurement):
        """
        Two fridges push to different branches of the same repository, and a
        shared clone would have them fighting over the checked-out branch.

        """
        assert measurement.git_repo.endswith("-triton")

    def test_without_a_fridge_name_the_default_repository_is_used(
        self, tmp_path, station, live_registry, offline_live_server
    ):
        with patch.object(Measurement, "get_installed_packages", return_value=""):
            plain = Measurement(
                "W1", "hall-bar", "S1", "pinchoff", station, str(tmp_path), {}
            )

        assert plain.git_repo == os.path.expanduser("~/.measurement-hashes")


class TestGetInstalledPackages:
    """
    The environment freeze that goes into the dataset metadata.

    Called unbound, since every other test in this module patches the method
    out -- freezing an environment costs seconds per measurement.

    """

    freeze = staticmethod(Measurement.get_installed_packages)

    def test_prefers_uv(self):
        with patch.object(
            subprocess, "check_output", return_value="qcodes==0.54\n"
        ) as check:
            frozen = self.freeze(None)

        assert frozen == "qcodes==0.54\n"
        assert check.call_args_list[0].args[0][0] == "uv"

    def test_falls_back_to_pip_when_uv_is_absent(self):
        """uv-created environments need not contain pip, and vice versa."""
        outputs = [FileNotFoundError(), "qcodes==0.54\n"]

        with patch.object(subprocess, "check_output", side_effect=outputs) as check:
            frozen = self.freeze(None)

        assert frozen == "qcodes==0.54\n"
        assert check.call_args_list[1].args[0][1:] == ["-m", "pip", "freeze"]

    def test_returns_nothing_when_neither_works(self):
        """
        An unrecorded environment is a worse dataset, not a failed
        measurement.

        """
        with patch.object(
            subprocess,
            "check_output",
            side_effect=subprocess.CalledProcessError(1, "uv"),
        ):
            assert self.freeze(None) == ""


class TestDatasetAllocation:
    """The empty grid a run fills in, and the metadata attached to it."""

    def test_a_data_array_spans_every_swept_parameter(self, measurement, gates, signal):
        sweeps = [Sweep(gates.x, 0.0, 1.0, num=3), Sweep(gates.y, 0.0, 1.0, num=4)]

        array = measurement._make_dataarray(sweeps, signal)

        assert array.dims == ("x", "y")
        assert array.shape == (3, 4)

    def test_the_grid_starts_out_empty(self, measurement, gates, signal):
        """
        NaN rather than zero, so an interrupted run is visibly partial instead
        of looking like a measurement that read zero everywhere.

        """
        array = measurement._make_dataarray([Sweep(gates.x, 0.0, 1.0, num=3)], signal)

        assert np.isnan(array.values).all()

    def test_the_dependent_carries_its_unit_label_and_instrument(
        self, measurement, gates, signal
    ):
        array = measurement._make_dataarray([Sweep(gates.x, 0.0, 1.0, num=2)], signal)

        assert array.attrs == {
            "unit": "A",
            "label": "Signal",
            "instrument": gates.name,
        }

    def test_each_coordinate_carries_its_own_metadata(self, measurement, gates, signal):
        """
        Qimchi labels its axes from these, so a coordinate without them plots
        as a bare number.

        """
        array = measurement._make_dataarray([Sweep(gates.x, 0.0, 1.0, num=2)], signal)

        assert array.coords["x"].attrs["unit"] == "V"
        assert array.coords["x"].attrs["label"] == gates.x.label
        assert array.coords["x"].attrs["instrument"] == gates.name

    def test_a_dataset_holds_one_variable_per_dependent(
        self, measurement, gates, signal
    ):
        second = Parameter("other", unit="V", instrument=gates, get_cmd=lambda: 0.0)

        dataset = measurement._make_dataset(
            [Sweep(gates.x, 0.0, 1.0, num=3)], [signal, second]
        )

        assert set(dataset.data_vars) == {"signal", "other"}
        assert dataset.sizes == {"x": 3}

    def test_the_dataset_records_the_sample_hierarchy(self, measurement, gates, signal):
        dataset = measurement._make_dataset([Sweep(gates.x, 0.0, 1.0, num=2)], [signal])

        assert dataset.attrs["Wafer ID"] == "W1"
        assert dataset.attrs["Device Type"] == "hall-bar"
        assert dataset.attrs["Sample Name"] == "S1"
        assert dataset.attrs["Experiment Name"] == "pinchoff"
        assert dataset.attrs["Measurement ID"] == measurement.id

    def test_the_cryostat_is_taken_from_the_fridge_name(
        self, measurement, gates, signal
    ):
        measurement._make_dataset([Sweep(gates.x, 0.0, 1.0, num=2)], [signal])

        assert measurement.cryostat == "triton"

    def test_without_a_fridge_name_the_cryostat_falls_back_to_the_hostname(
        self, tmp_path, station, gates, signal, live_registry, offline_live_server
    ):
        """
        The hostname convention is ``<lab>-<fridge>``; a machine that does not
        follow it must still produce a dataset.

        """
        with patch.object(Measurement, "get_installed_packages", return_value=""):
            plain = Measurement(
                "W1", "hall-bar", "S1", "pinchoff", station, str(tmp_path), {}
            )

        plain._make_dataset([Sweep(gates.x, 0.0, 1.0, num=2)], [signal])

        assert isinstance(plain.cryostat, str)
        assert plain.cryostat

    def test_a_hostname_without_the_lab_prefix_falls_back_to_a_placeholder(
        self, measurement, gates, signal, monkeypatch
    ):
        """
        The convention is ``<lab>-<fridge>``. A laptop called ``standalone``
        must still produce a dataset, just one that cannot name its fridge.

        """
        measurement.fridge_name = ""
        monkeypatch.setattr("qcutils.measure.socket.gethostname", lambda: "standalone")

        measurement._make_dataset([Sweep(gates.x, 0.0, 1.0, num=2)], [signal])

        assert measurement.cryostat == "dummy"


class TestPrintTable:
    """The run summary printed before a measurement starts."""

    def test_columns_are_padded_to_a_common_width(self, measurement, capsys):
        measurement._print_table(
            [["up1", "Up Plunger 1", -1.0, "V"], ["d", "D", 0.5, "V"]],
            ["Name", "Label", "Value", "Unit"],
        )

        header, separator, first, second = capsys.readouterr().out.splitlines()
        assert len(header) == len(separator) == len(first) == len(second)

    def test_a_header_wider_than_its_column_still_fits(self, measurement, capsys):
        measurement._print_table([["a"]], ["Independent(s)"])

        header, _, row = capsys.readouterr().out.splitlines()
        assert row.startswith("a")
        assert len(row) == len(header)

    def test_non_string_cells_are_rendered(self, measurement, capsys):
        measurement._print_table([[3, 0.25]], ["Points", "Delay"])

        assert "0.25" in capsys.readouterr().out


class TestDataHash:
    """The content hash pushed to the measurement-hashes repository."""

    def test_hashes_a_netcdf_file(self, measurement, tmp_path):
        Path(measurement.data).write_bytes(b"measurement bytes")

        assert len(measurement._compute_data_hash()) == 64

    def test_the_hash_follows_the_content(self, measurement):
        Path(measurement.data).write_bytes(b"first")
        first = measurement._compute_data_hash()
        Path(measurement.data).write_bytes(b"second")

        assert measurement._compute_data_hash() != first

    def test_hashes_a_directory_dataset_too(self, measurement):
        """Older measurements were left as ``.zarr`` directories."""
        folder = Path(measurement.data)
        folder.mkdir()
        (folder / "chunk.0").write_bytes(b"data")

        assert len(measurement._compute_data_hash()) == 64


class TestFinalizeDiskArtifacts:
    """Turning the live zarr store into the single netCDF file that ships."""

    @pytest.fixture
    def seeded(self, measurement, gates, signal):
        """
        A measurement with a dataset and a live zarr store on disk, as it
        would be part-way through a run.

        Returns:
            xr.Dataset: The in-memory dataset.

        """
        dataset = measurement._make_dataset([Sweep(gates.x, 0.0, 1.0, num=3)], [signal])
        dataset["signal"].values[:] = [1.0, 2.0, 3.0]
        measurement.arr = dataset
        dataset.to_zarr(measurement.live_data, mode="w")
        return dataset

    def test_exports_the_dataset_and_removes_the_zarr_store(self, measurement, seeded):
        measurement._finalize_disk_artifacts(dataset=seeded)

        assert Path(measurement.data).exists()
        assert not Path(measurement.live_data).exists()
        np.testing.assert_allclose(
            xr.load_dataset(measurement.data)["signal"].values, [1.0, 2.0, 3.0]
        )

    def test_prefers_the_in_memory_dataset_over_the_zarr_store(
        self, measurement, seeded
    ):
        """
        A checkpoint can lag behind after a transient lock, so ``self.arr`` is
        the authoritative copy and must win.

        """
        seeded["signal"].values[:] = [9.0, 9.0, 9.0]

        measurement._finalize_disk_artifacts()

        np.testing.assert_allclose(
            xr.load_dataset(measurement.data)["signal"].values, [9.0, 9.0, 9.0]
        )

    def test_falls_back_to_the_zarr_store_when_nothing_is_in_memory(
        self, measurement, seeded
    ):
        measurement.arr = None

        measurement._finalize_disk_artifacts()

        np.testing.assert_allclose(
            xr.load_dataset(measurement.data)["signal"].values, [1.0, 2.0, 3.0]
        )

    def test_with_nothing_to_export_it_reports_rather_than_raising(
        self, measurement, caplog
    ):
        """
        Finalisation runs in a ``finally``; raising here would replace the
        error that actually ended the run.

        """
        measurement.arr = None

        assert measurement._finalize_disk_artifacts() is None
        assert "No dataset available to export" in caplog.text

    def test_a_run_always_publishes_with_a_disk_path(
        self,
        measurement,
        gates,
        signal,
        live_registry,
        offline_live_server,
        quiet_registry_maintenance,
    ):
        """
        Qimchi Connect can record a measurement without a disk path, so QCUtils
        must guarantee its own runs have one. The run seeds a ``.zarr`` store
        before publishing and advertises that path.

        """
        published: list = []
        real = measure._register_memory_store

        def capture(measurement_id, store, disk_path=None):
            # Recorded here rather than after the run: finalisation removes the
            # live .zarr store once the netCDF export succeeds.
            published.append(
                (measurement_id, disk_path, disk_path and Path(disk_path).exists())
            )
            return real(measurement_id, store, disk_path=disk_path)

        with patch.object(measure, "_register_memory_store", capture):
            measurement.run(Sweep(gates.x, 0.0, 1.0, num=2), [signal], no_hashing=True)

        assert len(published) == 1
        measurement_id, disk_path, existed = published[0]
        assert measurement_id == measurement.id
        assert disk_path == str(Path(measurement.live_data).resolve())
        assert existed, "the .zarr store is seeded before the measurement publishes"

    def test_a_running_measurement_is_discoverable_with_a_usable_path(
        self,
        measurement,
        gates,
        signal,
        live_registry,
        offline_live_server,
        quiet_registry_maintenance,
    ):
        """
        While a run is active, Qimchi reads the registry row to locate it. The
        row must carry the live ``.zarr`` path as soon as it is written, rather
        than receiving only the final netCDF path during finalisation.

        """
        rows: list = []
        real = measure._register_memory_store

        def capture(measurement_id, store, disk_path=None):
            port = real(measurement_id, store, disk_path=disk_path)
            rows.append(live_db.get_measurement(measurement_id))
            return port

        with patch.object(measure, "_register_memory_store", capture):
            measurement.run(Sweep(gates.x, 0.0, 1.0, num=2), [signal], no_hashing=True)

        assert len(rows) == 1
        assert rows[0] is not None
        assert rows[0].live_status is True
        assert rows[0].fpath == str(Path(measurement.live_data).resolve())

    def test_the_live_database_is_pointed_at_the_final_file(
        self,
        measurement,
        seeded,
        live_registry,
        offline_live_server,
        quiet_registry_maintenance,
    ):
        _register_memory_store(
            measurement.id, zarr.MemoryStore(), disk_path=measurement.live_data
        )

        measurement._finalize_disk_artifacts(dataset=seeded)

        record = live_db.get_measurement(measurement.id)
        assert record.fpath == str(Path(measurement.data).resolve())
        assert get_live_registration(measurement.id).disk_path == record.fpath

    def test_a_failed_export_rescues_the_live_store_beside_the_data(
        self, measurement, seeded, caplog
    ):
        """
        The live store is the only copy left when the export fails, so it has
        to end up somewhere the user will find it.

        """
        with (
            patch.object(
                xr.Dataset, "to_netcdf", side_effect=RuntimeError("disk full")
            ),
            patch.object(
                measure, "_convert_live_store", side_effect=RuntimeError("disk full")
            ),
        ):
            assert measurement._finalize_disk_artifacts(dataset=seeded) is None

        rescued = Path(measurement.datalogging) / f"{measurement.id}.zarr"
        assert rescued.exists()
        assert not Path(measurement.live_data).exists()
        assert "very likely incomplete" in caplog.text

    def test_an_undeletable_zarr_store_does_not_fail_the_export(
        self, measurement, seeded, monkeypatch, caplog
    ):
        monkeypatch.setattr(
            "qcutils.measure.shutil.rmtree",
            lambda *a, **k: (_ for _ in ()).throw(OSError("locked")),
        )

        measurement._finalize_disk_artifacts(dataset=seeded)

        assert Path(measurement.data).exists()
        assert "Failed removing temporary zarr store" in caplog.text


class TestPushGitlab:
    """
    The hash push, exercised against a local bare repository.

    The real remote is a GitLab instance reached over SSH, but nothing in the
    function is specific to it: pointing ``origin`` at a bare repository on
    disk exercises the clone-or-open, branch-per-cryostat and commit logic for
    real.

    """

    @pytest.fixture
    def hashes_repo(self, tmp_path, measurement):
        """
        An initialised hash repository with a local bare origin.

        Returns:
            Repo: The working clone ``_push_gitlab`` will operate on.

        """
        origin = tmp_path / "hashes-origin.git"
        Repo.init(origin, bare=True, initial_branch="main")
        work = tmp_path / "hashes"
        repo = Repo.init(work)
        with repo.config_writer() as config:
            config.set_value("user", "name", "qcutils tests")
            config.set_value("user", "email", "tests@example.invalid")
        repo.create_remote("origin", str(origin))

        measurement.git_repo = str(work)
        measurement.cryostat = "triton"
        return repo

    def test_writes_the_hash_under_the_sample_hierarchy(self, measurement, hashes_repo):
        dataset = xr.Dataset(attrs={"Measurement ID": measurement.id})

        measurement._push_gitlab(dataset, "abc123")

        written = (
            Path(measurement.git_repo)
            / "W1"
            / "hall-bar"
            / "S1"
            / "pinchoff"
            / measurement.id
        )
        assert written.read_text().startswith("Hash: abc123")

    def test_records_the_dataset_attributes_alongside_the_hash(
        self, measurement, hashes_repo
    ):
        """
        The hash alone proves nothing about what was measured; the attributes
        are what make the record identifiable later.

        """
        dataset = xr.Dataset(attrs={"Sample Name": "S1", "Wafer ID": "W1"})

        measurement._push_gitlab(dataset, "abc123")

        written = (
            Path(measurement.git_repo) / "W1/hall-bar/S1/pinchoff" / measurement.id
        ).read_text()
        assert json.loads(written.split("\n\n", 1)[1])["Sample Name"] == "S1"

    def test_commits_onto_a_branch_named_for_the_cryostat(
        self, measurement, hashes_repo
    ):
        """One branch per fridge, so two fridges never collide on a push."""
        measurement._push_gitlab(xr.Dataset(), "abc123")

        assert hashes_repo.active_branch.name == "triton"
        assert "triton" in [
            head.name for head in Repo(hashes_repo.remotes.origin.url).heads
        ]

    def test_a_second_measurement_reuses_the_existing_branch(
        self, measurement, hashes_repo
    ):
        measurement._push_gitlab(xr.Dataset(), "abc123")
        measurement.id = "2-second"

        measurement._push_gitlab(xr.Dataset(), "def456")

        assert hashes_repo.active_branch.name == "triton"
        assert len(list(hashes_repo.iter_commits())) == 2

    def test_a_branch_that_only_exists_on_the_remote_is_checked_out(
        self, measurement, hashes_repo
    ):
        """
        The second machine measuring on a fridge finds the branch on the
        remote but not locally, and must track it rather than starting a
        parallel history.

        """
        measurement._push_gitlab(xr.Dataset(), "abc123")
        hashes_repo.git.checkout("-b", "scratch")
        hashes_repo.git.branch("-D", "triton")
        measurement.id = "2-second"

        measurement._push_gitlab(xr.Dataset(), "def456")

        assert hashes_repo.active_branch.name == "triton"
        assert hashes_repo.active_branch.tracking_branch() is not None

    def test_an_unchanged_hash_does_not_produce_an_empty_commit(
        self, measurement, hashes_repo, caplog
    ):
        measurement._push_gitlab(xr.Dataset(), "abc123")

        measurement._push_gitlab(xr.Dataset(), "abc123")

        assert "No changes to commit" in caplog.text
        assert len(list(hashes_repo.iter_commits())) == 1


class TestRun:
    """End-to-end measurements, from an empty grid to a netCDF file."""

    def test_a_one_dimensional_sweep_is_measured_and_exported(
        self, measurement, gates, signal
    ):
        swept = Sweep(gates.x, 0.0, 1.0, num=5)

        measurement.run([swept], [signal], no_hashing=True)

        exported = xr.load_dataset(measurement.data)
        np.testing.assert_allclose(exported["signal"].values, 100.0 * swept.values)

    def test_a_two_dimensional_sweep_lands_on_the_expected_grid(
        self, measurement, gates, signal
    ):
        outer = Sweep(gates.x, 0.0, 2.0, num=3)
        inner = Sweep(gates.y, 0.0, 1.0, num=2)

        measurement.run([outer, inner], [signal], no_hashing=True)

        exported = xr.load_dataset(measurement.data)
        expected = 100.0 * outer.values[:, None] + inner.values[None, :]
        np.testing.assert_allclose(exported["signal"].values, expected)

    def test_a_bare_sweep_does_not_have_to_be_wrapped_in_a_list(
        self, measurement, gates, signal
    ):
        measurement.run(Sweep(gates.x, 0.0, 1.0, num=3), [signal], no_hashing=True)

        assert xr.load_dataset(measurement.data)["signal"].notnull().all()

    def test_the_live_zarr_store_is_cleaned_up(self, measurement, gates, signal):
        measurement.run(Sweep(gates.x, 0.0, 1.0, num=3), [signal], no_hashing=True)

        assert not Path(measurement.live_data).exists()

    def test_the_measurement_is_unregistered_when_it_finishes(
        self, measurement, gates, signal
    ):
        measurement.run(Sweep(gates.x, 0.0, 1.0, num=3), [signal], no_hashing=True)

        assert get_live_registration(measurement.id) is None

    def test_the_exported_dataset_carries_the_run_metadata(
        self, measurement, gates, signal
    ):
        """
        Everything needed to reproduce the run: the instrument state, the
        station parameters, the sweeps and the user's own metadata.

        """
        measurement.run(Sweep(gates.x, 0.0, 1.0, num=3), [signal], no_hashing=True)

        attrs = xr.load_dataset(measurement.data).attrs
        assert json.loads(attrs["Extra Metadata"]) == {"operator": "test"}
        assert gates.name in json.loads(attrs["Instruments Snapshot"])
        assert "x_gate" in json.loads(attrs["Parameters Snapshot"])
        assert json.loads(attrs["Sweeps"])["Gate x"]["Number of Points"] == 3
        assert attrs["Requirements"] == "qcutils==0.8.0\n"

    def test_a_snapshot_that_is_not_serialisable_is_sanitised(
        self, measurement, gates, signal
    ):
        """
        Drivers leave context objects in their parameter caches after a sweep.
        The whole metadata blob must survive one of them.

        """
        gates.snapshot = lambda *a, **k: {"context": object()}

        measurement.run(Sweep(gates.x, 0.0, 1.0, num=2), [signal], no_hashing=True)

        snapshot = json.loads(
            xr.load_dataset(measurement.data).attrs["Instruments Snapshot"]
        )
        assert isinstance(snapshot[gates.name]["context"], str)

    def test_a_long_parameter_value_is_truncated_in_the_summary(
        self, tmp_path, gates, live_registry, offline_live_server
    ):
        """
        Array-valued parameters would otherwise make the printed table
        unreadable and bloat the attributes.

        """
        waveform = Parameter(
            "waveform", unit="V", instrument=gates, get_cmd=lambda: np.arange(100)
        )
        scalar = Parameter("signal", unit="A", instrument=gates, get_cmd=lambda: 1.0)
        built = Station("s")
        built.instruments.append(gates)
        built.add_parameter("waveform", "Waveform", waveform)
        with patch.object(Measurement, "get_installed_packages", return_value=""):
            long_value = Measurement(
                "W1", "hall-bar", "S1", "pinchoff", built, str(tmp_path), {}
            )

        long_value.run(Sweep(gates.y, 0.0, 1.0, num=2), [scalar], no_hashing=True)

        recorded = json.loads(
            xr.load_dataset(long_value.data).attrs["Parameters Snapshot"]
        )
        assert recorded["waveform"]["value"].endswith("...")
        assert len(recorded["waveform"]["value"]) == 53

    def test_prints_the_parameter_and_sweep_summaries(
        self, measurement, gates, signal, capsys
    ):
        measurement.run(Sweep(gates.x, 0.0, 1.0, num=2), [signal], no_hashing=True)

        printed = capsys.readouterr().out
        assert "Independent(s)" in printed
        assert "x_gate" in printed

    def test_an_interrupt_still_exports_what_was_measured(
        self, measurement, gates, signal
    ):
        """
        An interrupted run is the common case -- a user stopping a sweep that
        has already shown what it needed to -- and its data must survive.

        """
        readings = {"count": 0}

        def stop_after_two():
            readings["count"] += 1
            return readings["count"] > 2

        with pytest.raises(InterruptedError):
            measurement.run(
                Sweep(gates.x, 0.0, 1.0, num=10),
                [signal],
                interrupt=stop_after_two,
                no_hashing=True,
            )

        exported = xr.load_dataset(measurement.data)["signal"].values
        assert not np.isnan(exported[:2]).any()
        assert np.isnan(exported[-1])
        assert get_live_registration(measurement.id) is None

    def test_a_failing_dependent_ends_the_run_and_exports_the_partial_grid(
        self, measurement, gates
    ):
        readings = iter([1.0, 2.0])
        # snapshot_get=False so the station snapshot in `run` does not consume a
        # reading -- only the sweep should draw from the iterator.
        flaky = Parameter(
            "signal",
            unit="A",
            instrument=gates,
            get_cmd=lambda: next(readings),
            snapshot_get=False,
        )

        with pytest.raises(StopIteration):
            measurement.run(Sweep(gates.x, 0.0, 1.0, num=5), [flaky], no_hashing=True)

        exported = xr.load_dataset(measurement.data)["signal"].values
        np.testing.assert_allclose(exported[:2], [1.0, 2.0])

    def test_an_incomplete_sweep_is_reported_rather_than_exported_as_finished(
        self, measurement, gates, signal
    ):
        """
        The point count is the one check that a sweep really visited every
        cell. Without it a silently truncated run looks like a complete one.

        """
        with patch(
            "qcutils.measure._stepper", side_effect=lambda dataset, **kw: dataset
        ):
            with pytest.raises(RuntimeError, match="0/3 points acquired"):
                measurement.run(
                    Sweep(gates.x, 0.0, 1.0, num=3), [signal], no_hashing=True
                )

    def test_hashing_is_attempted_when_it_is_not_disabled(
        self, measurement, gates, signal
    ):
        with patch.object(Measurement, "_push_gitlab") as push:
            measurement.run(Sweep(gates.x, 0.0, 1.0, num=2), [signal])

        dataset, data_hash = push.call_args.args
        assert len(data_hash) == 64
        assert dataset.attrs["Measurement ID"] == measurement.id

    def test_a_failed_hash_push_does_not_fail_the_measurement(
        self, measurement, gates, signal, caplog
    ):
        """
        The data is already on disk by then; an unreachable remote is a
        deferred push, not a lost run.

        """
        with patch.object(
            Measurement, "_push_gitlab", side_effect=RuntimeError("no network")
        ):
            measurement.run(Sweep(gates.x, 0.0, 1.0, num=2), [signal])

        assert "Did not push measurement hash" in caplog.text
        assert Path(measurement.data).exists()

    def test_a_failed_export_is_reported(
        self, measurement, gates, signal, monkeypatch, caplog
    ):
        monkeypatch.setattr(
            xr.Dataset,
            "to_netcdf",
            lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
        )

        measurement.run(Sweep(gates.x, 0.0, 1.0, num=2), [signal], no_hashing=True)

        assert "Failed exporting final dataset" in caplog.text
        assert not Path(measurement.data).exists()

    def test_a_failed_export_skips_hashing(
        self, measurement, gates, signal, monkeypatch, caplog
    ):
        """
        There is nothing to hash if nothing was written, and the run should
        say so rather than dying inside the hash step.

        """
        monkeypatch.setattr(
            xr.Dataset,
            "to_netcdf",
            lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
        )

        with patch.object(Measurement, "_push_gitlab") as push:
            measurement.run(Sweep(gates.x, 0.0, 1.0, num=2), [signal])

        assert "final netCDF export failed" in caplog.text
        push.assert_not_called()

    def test_an_interrupt_can_ramp_the_gates_down(self, measurement, gates):
        def interrupted():
            raise KeyboardInterrupt

        dependent = Parameter("signal", unit="A", instrument=gates, get_cmd=interrupted)

        with pytest.raises(KeyboardInterrupt):
            measurement.run(
                Sweep(gates.x, 0.5, 1.0, num=3),
                [dependent],
                rampdown_on_interrupt=True,
                no_hashing=True,
            )

        assert gates.x() == 0.0


class TestRunHelper:
    """The one-call wrapper around building and running a Measurement."""

    def test_requires_a_station(self, gates, signal):
        """
        Without a station nothing is snapshotted, and the dataset would record
        no instrument state at all.

        """
        with pytest.raises(ValueError, match="Station is required"):
            run(
                Sweep(gates.x, 0.0, 1.0, num=2),
                [signal],
                wafer_id="W1",
                device_type="hall-bar",
                sample_name="S1",
                experiment_name="pinchoff",
            )

    def test_runs_a_measurement_and_can_return_its_location(
        self, tmp_path, station, gates, signal, live_registry, offline_live_server
    ):
        with patch.object(Measurement, "get_installed_packages", return_value=""):
            location = run(
                Sweep(gates.x, 0.0, 1.0, num=3),
                [signal],
                wafer_id="W1",
                device_type="hall-bar",
                sample_name="S1",
                experiment_name="pinchoff",
                station=station,
                data_location=str(tmp_path / "data"),
                location_return=True,
                no_hashing=True,
            )

        assert xr.load_dataset(location)["signal"].notnull().all()

    def test_returns_nothing_unless_the_location_is_asked_for(
        self, tmp_path, station, gates, signal, live_registry, offline_live_server
    ):
        with patch.object(Measurement, "get_installed_packages", return_value=""):
            result = run(
                Sweep(gates.x, 0.0, 1.0, num=2),
                [signal],
                wafer_id="W1",
                device_type="hall-bar",
                sample_name="S1",
                experiment_name="pinchoff",
                station=station,
                data_location=str(tmp_path / "data"),
                no_hashing=True,
            )

        assert result is None


class FakeBufferedSweepNode:
    """
    A buffered sweep node: it takes a coordinate array and triggers on it.

    Attributes:
        runs (int): Number of times the block was started.

    """

    def __init__(self) -> None:
        self.toplevel = False
        self.runs = 0

    def register_sweep(self, sweep, **kwargs):
        """
        Accept the sweep and report its geometry.

        Args:
            sweep (Sequence): Sweeps this node will drive.
            **kwargs: Trigger configuration, ignored here.

        Returns:
            tuple[str, int, float]: Trigger type, point count and spacing.

        """
        return "step", len(sweep[0].values), sweep[0].delay

    def run_sweep(self):
        self.runs += 1

    def abort(self):
        pass


class FakeBufferedAcquisitionNode:
    """
    A buffered acquisition node returning one block per fetch.

    Attributes:
        block (np.ndarray): What each fetch returns.
        fetches (int): Number of completed fetches.

    """

    def __init__(self, block) -> None:
        self.block = block
        self.fetches = 0

    def register_dependent(self, dependent, num, delay, **kwargs):
        self.dependents = dependent

    def fetch(self):
        self.fetches += 1
        return [self.block + 10.0 * self.fetches]

    def abort(self):
        pass


class TestBufferedRun:
    """
    A measurement whose innermost dimension is swept by the instrument itself.

    The slow sweep is stepped point by point in Python; at each of its points
    the buffered tree is armed, run once, and read back as a whole block. The
    dataset therefore has one dimension per slow sweep and one per buffered
    sweep, and the block has to land on the right slice of it.

    """

    @pytest.fixture
    def buffered(self, gates):
        """
        A one-deep buffered tree over ``gates.y``, with one dependent.

        Returns:
            SimpleNamespace: The payload plus the nodes and dependent in it.

        """
        root = FakeBufferedSweepNode()
        reader = FakeBufferedAcquisitionNode(np.arange(4.0))
        dependent = Parameter(
            "buffered_signal", unit="A", label="Buffered signal", instrument=gates
        )
        return SimpleNamespace(
            root=root,
            reader=reader,
            dependent=dependent,
            payload={
                "instrument": root,
                "sweeps": [Sweep(gates.y, 0.0, 1.0, num=4, delay=0.0)],
                "nodes": [{"instrument": reader, "dependent": [dependent]}],
            },
        )

    def test_the_dataset_spans_the_slow_and_buffered_dimensions(
        self, measurement, gates, buffered
    ):
        measurement.run(
            [Sweep(gates.x, 0.0, 1.0, num=2), buffered.payload], [], no_hashing=True
        )

        exported = xr.load_dataset(measurement.data)
        assert exported["buffered_signal"].dims == ("x", "y")
        assert exported.sizes == {"x": 2, "y": 4}

    def test_each_block_lands_on_its_own_slow_point(self, measurement, gates, buffered):
        """
        The fake returns a different block per fetch, so a block written to
        the wrong slice -- or written twice -- is visible in the values.

        """
        measurement.run(
            [Sweep(gates.x, 0.0, 1.0, num=2), buffered.payload], [], no_hashing=True
        )

        exported = xr.load_dataset(measurement.data)["buffered_signal"].values
        np.testing.assert_allclose(exported[0], np.arange(4.0) + 10.0)
        np.testing.assert_allclose(exported[1], np.arange(4.0) + 20.0)

    def test_the_tree_is_re_armed_and_re_run_at_every_slow_point(
        self, measurement, gates, buffered
    ):
        """
        Buffered instruments can only be read once per arming, so the block
        has to be set up again at each slow step rather than started once.

        """
        measurement.run(
            [Sweep(gates.x, 0.0, 1.0, num=3), buffered.payload], [], no_hashing=True
        )

        assert buffered.root.runs == 3
        assert buffered.reader.fetches == 3

    def test_the_buffered_sweep_appears_in_the_run_metadata(
        self, measurement, gates, buffered
    ):
        """
        The buffered dimension is as much part of the measurement as the slow
        one, and a dataset that only records the slow sweep is unreadable.

        """
        measurement.run(
            [Sweep(gates.x, 0.0, 1.0, num=2), buffered.payload], [], no_hashing=True
        )

        sweeps = json.loads(xr.load_dataset(measurement.data).attrs["Sweeps"])
        assert set(sweeps) == {"Gate x", "Gate y"}
        assert sweeps["Gate y"]["Number of Points"] == 4

    def test_a_purely_buffered_run_needs_no_slow_sweep(
        self, measurement, gates, buffered
    ):
        measurement.run([buffered.payload], [], no_hashing=True)

        exported = xr.load_dataset(measurement.data)
        assert exported["buffered_signal"].dims == ("y",)
        np.testing.assert_allclose(
            exported["buffered_signal"].values, np.arange(4.0) + 10.0
        )

    def test_an_interrupt_aborts_the_buffered_instruments(
        self, measurement, gates, buffered
    ):
        """
        Instruments left armed keep triggering after the run ends, so Ctrl-C
        has to reach them.

        """
        aborted = []
        buffered.reader.abort = lambda: aborted.append("reader")
        buffered.root.abort = lambda: aborted.append("root")
        buffered.root.run_sweep = lambda: (_ for _ in ()).throw(KeyboardInterrupt)

        with pytest.raises(KeyboardInterrupt):
            measurement.run([buffered.payload], [], no_hashing=True)

        assert set(aborted) == {"root", "reader"}
