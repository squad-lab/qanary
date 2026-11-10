"""Tests for non-buffered sweep execution and checkpointing."""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
import zarr
from qcodes.parameters import Parameter

from qanary import sweep as sweep_module
from qanary.sweep import PointSweep, Sweep, _stepper


@pytest.fixture
def signal(gates):
    """Return a dependent whose value identifies its grid position."""
    return Parameter(
        "signal",
        unit="A",
        label="Signal",
        instrument=gates,
        get_cmd=lambda: 100.0 * gates.x() + gates.y(),
    )


def make_dataset(*sweeps, name: str = "signal") -> xr.Dataset:
    """Preallocate a measurement dataset for the given sweeps."""
    dims = [param.name for sweep in sweeps for param in sweep.parameter]
    coords = {param.name: sweep.values for sweep in sweeps for param in sweep.parameter}
    shape = tuple(len(coords[dim]) for dim in dims)
    return xr.Dataset(
        {name: (dims, np.full(shape, np.nan))},
        coords=coords,
    )


# `_persist_disk` writes directly to this path when no disk store is supplied.
# Keep it under pytest's temporary directory.
_FALLBACK_LOCATION = None


@pytest.fixture(autouse=True)
def _fallback_location(tmp_path):
    """Put fallback checkpoints under pytest's temporary directory."""
    global _FALLBACK_LOCATION
    _FALLBACK_LOCATION = str(tmp_path / "stepper-fallback.zarr")
    yield
    _FALLBACK_LOCATION = None


def step(dataset, sweeps, dependents, bar, **kwargs):
    """Run ``_stepper`` with normal measurement defaults."""
    independents = [param for sweep in sweeps for param in sweep.parameter]
    return _stepper(
        dataset=dataset,
        data_location=kwargs.pop("data_location", _FALLBACK_LOCATION),
        depth=len(sweeps),
        sweeps=sweeps,
        independents=independents,
        dependents=dependents,
        sweep_cache=[0.0] * len(independents),
        bar=bar,
        **kwargs,
    )


class TestGridFilling:
    def test_a_one_dimensional_sweep_fills_every_point(self, gates, signal, bar):
        swept = Sweep(gates.x, 0.0, 1.0, num=5)
        dataset = make_dataset(swept)

        step(dataset, [swept], [signal], bar)

        np.testing.assert_allclose(dataset["signal"].values, 100.0 * swept.values)
        assert bar.n == 5

    def test_repeated_coordinates_keep_each_reading(self, gates, bar):
        readings = iter([10.0, 20.0, 30.0, 40.0])
        dependent = Parameter(
            "signal", unit="A", instrument=gates, get_cmd=lambda: next(readings)
        )
        swept = PointSweep(gates.x, [0.0, 1.0, 0.0, 1.0])
        dataset = make_dataset(swept)

        step(dataset, [swept], [dependent], bar)

        np.testing.assert_allclose(dataset["signal"].values, [10.0, 20.0, 30.0, 40.0])
        assert bar.n == 4

    def test_repeated_outer_coordinates_keep_their_grid_positions(self, gates, bar):
        readings = iter(np.arange(1.0, 7.0))
        dependent = Parameter(
            "signal", unit="A", instrument=gates, get_cmd=lambda: next(readings)
        )
        outer = PointSweep(gates.x, [0.0, 1.0, 0.0])
        inner = Sweep(gates.y, 0.0, 1.0, num=2)
        dataset = make_dataset(outer, inner)

        step(dataset, [outer, inner], [dependent], bar)

        np.testing.assert_allclose(
            dataset["signal"].values, np.arange(1.0, 7.0).reshape(3, 2)
        )
        assert bar.n == 6

    def test_repeated_points_shared_by_parameters_fill_the_diagonal(self, gates, bar):
        readings = iter([10.0, 20.0, 30.0])
        dependent = Parameter(
            "signal", unit="A", instrument=gates, get_cmd=lambda: next(readings)
        )
        swept = PointSweep([gates.x, gates.y], [0.0, 1.0, 0.0])
        dataset = make_dataset(swept)

        step(dataset, [swept], [dependent], bar)

        np.testing.assert_allclose(
            dataset["signal"].values,
            [[10.0, np.nan, np.nan], [np.nan, 20.0, np.nan], [np.nan, np.nan, 30.0]],
        )
        assert bar.n == 3

    def test_a_two_dimensional_sweep_lands_on_the_expected_cells(
        self, gates, signal, bar
    ):
        outer = Sweep(gates.x, 0.0, 2.0, num=3)
        inner = Sweep(gates.y, 0.0, 1.0, num=2)
        dataset = make_dataset(outer, inner)

        step(dataset, [outer, inner], [signal], bar)

        expected = 100.0 * outer.values[:, None] + inner.values[None, :]
        np.testing.assert_allclose(dataset["signal"].values, expected)
        assert bar.n == 6

    def test_a_three_dimensional_sweep_recurses_to_the_innermost_level(
        self, instrument, bar
    ):
        inst = instrument("a", "b", "c")
        dependent = Parameter(
            "signal",
            unit="A",
            label="Signal",
            instrument=inst,
            get_cmd=lambda: 100.0 * inst.a() + 10.0 * inst.b() + inst.c(),
        )
        sweeps = [
            Sweep(inst.a, 0.0, 1.0, num=2),
            Sweep(inst.b, 0.0, 1.0, num=2),
            Sweep(inst.c, 0.0, 1.0, num=2),
        ]
        dataset = make_dataset(*sweeps)

        step(dataset, sweeps, [dependent], bar)

        assert not np.isnan(dataset["signal"].values).any()
        assert bar.n == 8
        assert dataset["signal"].sel(a=1.0, b=1.0, c=1.0) == pytest.approx(111.0)

    def test_reads_every_dependent_at_every_point(self, gates, signal, bar):
        second = Parameter(
            "other", unit="V", instrument=gates, get_cmd=lambda: -gates.x()
        )
        swept = Sweep(gates.x, 0.0, 1.0, num=4)
        dataset = make_dataset(swept)
        dataset["other"] = (("x",), np.full(4, np.nan))

        step(dataset, [swept], [signal, second], bar)

        np.testing.assert_allclose(dataset["other"].values, -swept.values)
        assert bar.n == 4, "the bar counts points, not readings"

    def test_several_parameters_on_one_sweep_share_a_dimension_each(
        self, gates, signal, bar
    ):
        swept = Sweep([gates.x, gates.y], 0.0, 1.0, num=3)
        dataset = make_dataset(swept)

        step(dataset, [swept], [signal], bar)

        diagonal = np.diagonal(dataset["signal"].values)
        np.testing.assert_allclose(diagonal, 101.0 * swept.values)


class TestSweepMechanics:
    def test_start_delay_is_applied_once_per_sweep_entry(
        self, gates, signal, bar, monkeypatch
    ):
        dwells = []
        monkeypatch.setattr("qanary.sweep.sleep", dwells.append)
        outer = Sweep(gates.x, 0.0, 1.0, num=3, start_delay=5.0)
        inner = Sweep(gates.y, 0.0, 1.0, num=2, start_delay=0.5)
        dataset = make_dataset(outer, inner)

        step(dataset, [outer, inner], [signal], bar)

        assert dwells.count(5.0) == 1
        assert dwells.count(0.5) == 3

    def test_delay_is_applied_at_every_point(self, gates, signal, bar, monkeypatch):
        dwells = []
        monkeypatch.setattr("qanary.sweep.sleep", dwells.append)
        swept = Sweep(gates.x, 0.0, 1.0, num=4, delay=0.25)
        dataset = make_dataset(swept)

        step(dataset, [swept], [signal], bar)

        assert dwells.count(0.25) == 4

    def test_an_interrupt_stops_the_sweep(self, gates, signal, bar):
        swept = Sweep(gates.x, 0.0, 1.0, num=10)
        dataset = make_dataset(swept)

        with pytest.raises(InterruptedError):
            step(dataset, [swept], [signal], bar, interrupt=lambda: bar.n >= 3)

        assert np.isnan(dataset["signal"].values).any(), "it stopped early"
        assert not np.isnan(dataset["signal"].values[:3]).any(), "it kept its data"

    def test_an_interrupt_before_the_first_point_acquires_nothing(
        self, gates, signal, bar
    ):
        swept = Sweep(gates.x, 0.0, 1.0, num=5)
        dataset = make_dataset(swept)

        with pytest.raises(InterruptedError):
            step(dataset, [swept], [signal], bar, interrupt=lambda: True)

        assert np.isnan(dataset["signal"].values).all()

    def test_parallel_sweep_still_visits_every_point(self, gates, signal, bar):
        swept = Sweep([gates.x, gates.y], 0.0, 1.0, num=3)
        dataset = make_dataset(swept)

        step(dataset, [swept], [signal], bar, parallel_sweep=True)

        assert bar.n == 3

    def test_an_independent_missing_from_the_list_is_skipped(
        self, gates, signal, bar, caplog, tmp_path
    ):
        swept = Sweep(gates.x, 0.0, 1.0, num=2)
        dataset = make_dataset(swept)

        _stepper(
            dataset=dataset,
            data_location=str(tmp_path / "stepper.zarr"),
            depth=1,
            sweeps=[swept],
            independents=[],
            dependents=[signal],
            sweep_cache=[],
            bar=bar,
        )

        assert "not found in independents list" in caplog.text
        assert bar.n == 2


class TestPersistence:
    @pytest.fixture
    def stores(self, tmp_path):
        path = tmp_path / "live.zarr"
        return SimpleNamespace(
            memory={},
            disk=zarr.storage.LocalStore(str(path)),
            path=path,
        )

    def test_the_final_checkpoint_writes_the_complete_dataset(
        self, gates, signal, bar, stores
    ):
        swept = Sweep(gates.x, 0.0, 1.0, num=4)
        dataset = make_dataset(swept)

        step(
            dataset,
            [swept],
            [signal],
            bar,
            memory_store=stores.memory,
            disk_store=stores.disk,
        )

        restored = xr.open_zarr(stores.path, consolidated=False).load()
        np.testing.assert_allclose(restored["signal"].values, 100.0 * swept.values)

    def test_the_disk_store_matches_the_memory_store(self, gates, signal, bar, stores):
        swept = Sweep(gates.x, 0.0, 1.0, num=4)
        dataset = make_dataset(swept)

        step(
            dataset,
            [swept],
            [signal],
            bar,
            memory_store=stores.memory,
            disk_store=stores.disk,
        )

        written = {
            p.relative_to(stores.path).as_posix()
            for p in stores.path.rglob("*")
            if p.is_file()
        }
        assert written == set(stores.memory)
        for key, value in stores.memory.items():
            assert (stores.path / key).read_bytes() == value.to_bytes(), key

    def test_without_a_disk_store_the_dataset_is_written_to_the_location(
        self, gates, signal, bar, tmp_path
    ):
        location = tmp_path / "fallback.zarr"
        swept = Sweep(gates.x, 0.0, 1.0, num=3)
        dataset = make_dataset(swept)

        step(
            dataset,
            [swept],
            [signal],
            bar,
            data_location=str(location),
            memory_store={},
        )

        assert xr.open_zarr(location, consolidated=False)["signal"].notnull().all()

    def test_a_netcdf_snapshot_is_written_when_asked(
        self, gates, signal, bar, stores, tmp_path
    ):
        snapshot = tmp_path / "snapshot.nc"
        swept = Sweep(gates.x, 0.0, 1.0, num=3)
        dataset = make_dataset(swept)

        step(
            dataset,
            [swept],
            [signal],
            bar,
            memory_store=stores.memory,
            disk_store=stores.disk,
            nc_snapshot_path=str(snapshot),
        )

        assert snapshot.exists()
        assert not snapshot.with_suffix(".nc.tmp").exists()

    def test_a_failed_netcdf_snapshot_does_not_stop_the_sweep(
        self, gates, signal, bar, stores, tmp_path, caplog
    ):
        swept = Sweep(gates.x, 0.0, 1.0, num=3)
        dataset = make_dataset(swept)

        step(
            dataset,
            [swept],
            [signal],
            bar,
            memory_store=stores.memory,
            disk_store=stores.disk,
            nc_snapshot_path=str(tmp_path / "no-such-dir" / "snapshot.nc"),
        )

        assert bar.n == 3
        assert "Failed netCDF snapshot" in caplog.text

    def test_verbose_logging_reports_each_checkpoint(
        self, gates, signal, bar, stores, caplog
    ):
        swept = Sweep(gates.x, 0.0, 1.0, num=2)
        dataset = make_dataset(swept)

        step(
            dataset,
            [swept],
            [signal],
            bar,
            memory_store=stores.memory,
            disk_store=stores.disk,
            verbose=True,
        )

        assert "Saved dataset to in-memory store" in caplog.text
        assert "changed key(s)" in caplog.text

    def test_a_missing_memory_store_is_warned_about_not_fatal(
        self, gates, signal, bar, tmp_path, caplog
    ):
        swept = Sweep(gates.x, 0.0, 1.0, num=2)
        dataset = make_dataset(swept)

        step(
            dataset,
            [swept],
            [signal],
            bar,
            data_location=str(tmp_path / "no-memory.zarr"),
        )

        assert "No memory store provided" in caplog.text
        assert bar.n == 2


class TestLockedDiskStore:
    class LockedStore(zarr.storage.LocalStore):
        def __init__(self, root, failures: int) -> None:
            super().__init__(root)
            self.failures = failures
            self.attempts = 0
            self._failing_batch = -1

        async def set(self, key, value, *args, **kwargs):
            if self._failing_batch != self.attempts:
                self._failing_batch = self.attempts
                self.attempts += 1
                if self.attempts <= self.failures:
                    raise PermissionError("store is locked by another process")
            await super().set(key, value, *args, **kwargs)

    @pytest.fixture(autouse=True)
    def no_backoff(self, monkeypatch):
        monkeypatch.setattr("qanary.sweep.sleep", lambda seconds: None)

    def test_a_transient_lock_is_retried(self, gates, signal, bar, tmp_path, caplog):
        swept = Sweep(gates.x, 0.0, 1.0, num=3)
        dataset = make_dataset(swept)
        memory = {}
        disk = tmp_path / "locked.zarr"

        step(
            dataset,
            [swept],
            [signal],
            bar,
            memory_store=memory,
            disk_store=self.LockedStore(str(disk), failures=2),
        )

        assert "temporarily locked" in caplog.text
        assert bar.n == 3
        written = {
            p.relative_to(disk).as_posix() for p in disk.rglob("*") if p.is_file()
        }
        assert written == set(memory)

    def test_a_persistent_lock_does_not_lose_the_sweep(
        self, gates, signal, bar, tmp_path, caplog
    ):
        swept = Sweep(gates.x, 0.0, 1.0, num=3)
        dataset = make_dataset(swept)
        memory = {}

        step(
            dataset,
            [swept],
            [signal],
            bar,
            memory_store=memory,
            disk_store=self.LockedStore(str(tmp_path / "wedged.zarr"), failures=10_000),
        )

        assert "remains locked" in caplog.text
        assert bar.n == 3
        np.testing.assert_allclose(dataset["signal"].values, 100.0 * swept.values)
        assert xr.open_zarr(memory, consolidated=False)["signal"].notnull().all()


class TestFailureHandling:
    def test_a_failing_dependent_preserves_what_was_already_acquired(
        self, gates, bar, tmp_path
    ):
        readings = iter([1.0, 2.0])

        def flaky():
            return next(readings)

        dependent = Parameter("signal", unit="A", instrument=gates, get_cmd=flaky)
        swept = Sweep(gates.x, 0.0, 1.0, num=5)
        dataset = make_dataset(swept)
        memory = {}

        with pytest.raises(StopIteration):
            step(
                dataset,
                [swept],
                [dependent],
                bar,
                memory_store=memory,
                disk_store=zarr.storage.LocalStore(str(tmp_path / "partial.zarr")),
            )

        preserved = xr.open_zarr(memory, consolidated=False)["signal"].values
        np.testing.assert_allclose(preserved[:2], [1.0, 2.0])

    def test_a_keyboard_interrupt_preserves_the_in_memory_dataset(
        self, gates, bar, tmp_path
    ):
        readings = iter([7.0])

        def interrupted():
            try:
                return next(readings)
            except StopIteration:
                raise KeyboardInterrupt from None

        dependent = Parameter("signal", unit="A", instrument=gates, get_cmd=interrupted)
        swept = Sweep(gates.x, 0.0, 1.0, num=4)
        dataset = make_dataset(swept)
        memory = {}

        with pytest.raises(KeyboardInterrupt):
            step(dataset, [swept], [dependent], bar, memory_store=memory)

        preserved = xr.open_zarr(memory, consolidated=False)["signal"].values
        assert preserved[0] == pytest.approx(7.0)

    def test_a_failed_preservation_is_logged_and_the_original_error_wins(
        self, gates, signal, bar, caplog, monkeypatch
    ):

        def refuse(*args, **kwargs):
            raise RuntimeError("memory store is gone too")

        swept = Sweep(gates.x, 0.0, 1.0, num=2)
        dataset = make_dataset(swept)
        monkeypatch.setattr(type(dataset), "to_zarr", refuse)

        with pytest.raises(RuntimeError, match="memory store is gone too"):
            step(
                dataset,
                [swept],
                [signal],
                bar,
                memory_store={},
            )

        assert "Failed to preserve in-memory state" in caplog.text


class TestSaveInterval:
    def test_a_long_interval_still_checkpoints_at_the_end(
        self, gates, signal, bar, tmp_path
    ):
        swept = Sweep(gates.x, 0.0, 1.0, num=4)
        dataset = make_dataset(swept)
        memory = {}
        disk = zarr.storage.LocalStore(str(tmp_path / "throttled.zarr"))
        sweep_module.last_save = 1  # anything non-zero suppresses the first save

        step(
            dataset,
            [swept],
            [signal],
            bar,
            save_interval=3600.0,
            memory_store=memory,
            disk_store=disk,
        )

        restored = xr.open_zarr(disk, consolidated=False)["signal"].values
        np.testing.assert_allclose(restored, 100.0 * swept.values)


class TestDegenerateSweeps:
    def test_no_sweeps_reads_each_dependent_once(self, gates, signal, bar, tmp_path):
        gates.x(0.5)
        dataset = xr.Dataset({"signal": ((), np.nan)})

        _stepper(
            dataset=dataset,
            data_location=str(tmp_path / "stepper.zarr"),
            depth=0,
            sweeps=[],
            independents=[],
            dependents=[signal],
            sweep_cache=[],
            bar=bar,
            memory_store={},
        )

        assert dataset["signal"].item() == pytest.approx(50.0)
        assert bar.n == 1

    def test_a_disk_store_without_a_memory_store_is_written_directly(
        self, gates, signal, bar, tmp_path, caplog
    ):
        swept = Sweep(gates.x, 0.0, 1.0, num=3)
        dataset = make_dataset(swept)
        disk = zarr.storage.LocalStore(str(tmp_path / "direct.zarr"))

        step(dataset, [swept], [signal], bar, disk_store=disk, verbose=True)

        assert "directly to disk store" in caplog.text
        np.testing.assert_allclose(
            xr.open_zarr(disk, consolidated=False)["signal"].values,
            100.0 * swept.values,
        )
