"""
Tests for the non-buffered ``_stepper`` paths.

``_stepper`` is the recursion that turns a list of sweeps into nested loops:
at every level it drives the sweep parameters, and at the innermost level it
reads the dependents and writes them into the pre-allocated Xarray dataset.
The buffered paths are covered in ``test_buffered_progress.py``; this module
covers everything a run without hardware-buffered instruments goes through.

The assertions that matter most are about *where* a reading lands. The dataset
is allocated up front and indexed by coordinate value through ``sweep_cache``,
so an off-by-one in that bookkeeping does not crash -- it silently transposes
or shifts a measurement. Each test therefore uses a dependent whose value
identifies the point it was read at.

Checkpointing is the other concern. A sweep must survive a transiently locked
disk store, and must never lose the in-memory dataset when it fails.

"""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
import zarr
from qcodes.parameters import Parameter

from qanary import sweep as sweep_module
from qanary.sweep import Sweep, _stepper


@pytest.fixture
def signal(gates):
    """
    A dependent that encodes the gate positions it was read at.

    Returns:
        Parameter: Reads ``100 * x + y``, so every grid point is distinct.

    """
    return Parameter(
        "signal",
        unit="A",
        label="Signal",
        instrument=gates,
        get_cmd=lambda: 100.0 * gates.x() + gates.y(),
    )


def make_dataset(*sweeps, name: str = "signal") -> xr.Dataset:
    """
    Pre-allocate the dataset a measurement would seed before stepping.

    Args:
        *sweeps (Sweep): Sweeps in outer-to-inner order.
        name (str): Name of the single dependent variable.

    Returns:
        xr.Dataset: NaN-filled grid with one coordinate per swept parameter.

    """
    dims = [param.name for sweep in sweeps for param in sweep.parameter]
    coords = {param.name: sweep.values for sweep in sweeps for param in sweep.parameter}
    shape = tuple(len(coords[dim]) for dim in dims)
    return xr.Dataset(
        {name: (dims, np.full(shape, np.nan))},
        coords=coords,
    )


# Where `step` sends a checkpoint when the caller passes no disk store.
#
# `_persist_disk` falls back to `dataset.to_zarr(store=data_location)` in that
# case, writing `data_location` verbatim -- so a relative placeholder is not
# inert. The previous default, the string "unused", put a real Zarr store in the
# working directory, which under pytest is the repository root.
_FALLBACK_LOCATION = None


@pytest.fixture(autouse=True)
def _fallback_location(tmp_path):
    """Point the no-store checkpoint fallback at this test's tmp_path."""
    global _FALLBACK_LOCATION
    _FALLBACK_LOCATION = str(tmp_path / "stepper-fallback.zarr")
    yield
    _FALLBACK_LOCATION = None


def step(dataset, sweeps, dependents, bar, **kwargs):
    """
    Call ``_stepper`` with the arguments a measurement would supply.

    Args:
        dataset (xr.Dataset): Pre-allocated dataset.
        sweeps (list[Sweep]): Sweeps in outer-to-inner order.
        dependents (list[Parameter]): Parameters read at the innermost level.
        bar (Any): Progress bar.
        **kwargs: Passed straight through to ``_stepper``.

    Returns:
        xr.Dataset: Whatever ``_stepper`` returns.

    """
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
    """Every point read, and every point in the right cell."""

    def test_a_one_dimensional_sweep_fills_every_point(self, gates, signal, bar):
        swept = Sweep(gates.x, 0.0, 1.0, num=5)
        dataset = make_dataset(swept)

        step(dataset, [swept], [signal], bar)

        np.testing.assert_allclose(dataset["signal"].values, 100.0 * swept.values)
        assert bar.n == 5

    def test_a_two_dimensional_sweep_lands_on_the_expected_cells(
        self, gates, signal, bar
    ):
        """
        The inner sweep varies fastest, and the dependent encodes both gate
        positions, so a transposed or shifted assignment shows up immediately.

        """
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
        """
        A sweep over two parameters moves both together but still allocates a
        dimension per parameter, so the grid is the diagonal of a square.

        """
        swept = Sweep([gates.x, gates.y], 0.0, 1.0, num=3)
        dataset = make_dataset(swept)

        step(dataset, [swept], [signal], bar)

        diagonal = np.diagonal(dataset["signal"].values)
        np.testing.assert_allclose(diagonal, 101.0 * swept.values)


class TestSweepMechanics:
    """Dwells, interrupts and the parallel path."""

    def test_start_delay_is_applied_once_per_sweep_entry(
        self, gates, signal, bar, monkeypatch
    ):
        """
        ``start_delay`` lets an instrument settle after a jump back to the
        start of a sweep, so the inner sweep must pay it on every pass, not
        only on the first.

        """
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
        """
        The interrupt callable is polled before each point, and a raised
        InterruptedError is what ``Measurement.run`` catches to finalise the
        partial dataset.

        """
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
        """
        The parallel path sets each parameter on its own thread. The reading
        may race the setter, so this only asserts the grid was fully visited.

        """
        swept = Sweep([gates.x, gates.y], 0.0, 1.0, num=3)
        dataset = make_dataset(swept)

        step(dataset, [swept], [signal], bar, parallel_sweep=True)

        assert bar.n == 3

    def test_an_independent_missing_from_the_list_is_skipped(
        self, gates, signal, bar, caplog, tmp_path
    ):
        """
        ``slow_indexers`` is built by looking each swept parameter up in
        ``independents``. A parameter that is not there cannot be indexed, and
        the point is written to whatever the remaining indexers select rather
        than crashing mid-run.

        """
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
    """Checkpointing to the in-memory store, the disk store and netCDF."""

    @pytest.fixture
    def stores(self, tmp_path):
        """
        A memory store and a disk store, as ``Measurement.run`` seeds them.

        Returns:
            SimpleNamespace: ``memory`` and ``disk`` stores plus the disk path.

        """
        path = tmp_path / "live.zarr"
        return SimpleNamespace(
            memory=zarr.MemoryStore(),
            disk=zarr.DirectoryStore(str(path)),
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
        """
        The disk checkpoint copies only changed keys out of the memory store,
        so the two must end up byte-identical.

        """
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

        assert set(stores.disk.keys()) == set(stores.memory.keys())
        for key in stores.memory.keys():
            assert stores.disk[key] == stores.memory[key], key

    def test_without_a_disk_store_the_dataset_is_written_to_the_location(
        self, gates, signal, bar, tmp_path
    ):
        """The fallback path a measurement takes when no store was seeded."""
        location = tmp_path / "fallback.zarr"
        swept = Sweep(gates.x, 0.0, 1.0, num=3)
        dataset = make_dataset(swept)

        step(
            dataset,
            [swept],
            [signal],
            bar,
            data_location=str(location),
            memory_store=zarr.MemoryStore(),
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
        """
        The snapshot is explicitly best-effort -- it is known to fail on
        Windows while the zarr store is open -- so it must never take the run
        down with it.

        """
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
    """
    Windows can lock files while another process reads the advertised disk
    fallback. A checkpoint must retry, and if it still cannot write, it must let
    the sweep continue rather than throw away points already held in memory.

    """

    class LockedStore:
        """
        Disk-store proxy that refuses a fixed number of write attempts.

        Attributes:
            attempts (int): Number of write batches attempted so far.

        """

        def __init__(self, real, failures: int) -> None:
            self.real = real
            self.failures = failures
            self.attempts = 0
            self._failing_batch = -1

        def __contains__(self, key):
            return key in self.real

        def keys(self):
            return self.real.keys()

        def __getitem__(self, key):
            return self.real[key]

        def __setitem__(self, key, value):
            if self._failing_batch != self.attempts:
                self._failing_batch = self.attempts
                self.attempts += 1
                if self.attempts <= self.failures:
                    raise PermissionError("store is locked by another process")
            self.real[key] = value

    @pytest.fixture(autouse=True)
    def no_backoff(self, monkeypatch):
        """Skip the exponential backoff so the retries do not take seconds."""
        monkeypatch.setattr("qanary.sweep.sleep", lambda seconds: None)

    def test_a_transient_lock_is_retried(self, gates, signal, bar, tmp_path, caplog):
        swept = Sweep(gates.x, 0.0, 1.0, num=3)
        dataset = make_dataset(swept)
        memory = zarr.MemoryStore()
        disk = zarr.DirectoryStore(str(tmp_path / "locked.zarr"))

        step(
            dataset,
            [swept],
            [signal],
            bar,
            memory_store=memory,
            disk_store=self.LockedStore(disk, failures=2),
        )

        assert "temporarily locked" in caplog.text
        assert bar.n == 3
        assert set(disk.keys()) == set(memory.keys())

    def test_a_persistent_lock_does_not_lose_the_sweep(
        self, gates, signal, bar, tmp_path, caplog
    ):
        """
        After the retries are exhausted the checkpoint gives up, but the
        complete dataset is still in memory and the sweep carries on. A raised
        PermissionError here would discard an overnight run over a file
        handle.

        """
        swept = Sweep(gates.x, 0.0, 1.0, num=3)
        dataset = make_dataset(swept)
        memory = zarr.MemoryStore()
        disk = zarr.DirectoryStore(str(tmp_path / "wedged.zarr"))

        step(
            dataset,
            [swept],
            [signal],
            bar,
            memory_store=memory,
            disk_store=self.LockedStore(disk, failures=10_000),
        )

        assert "remains locked" in caplog.text
        assert bar.n == 3
        np.testing.assert_allclose(dataset["signal"].values, 100.0 * swept.values)
        assert xr.open_zarr(memory, consolidated=False)["signal"].notnull().all()


class TestFailureHandling:
    """What survives when a dependent or a parameter raises mid-sweep."""

    def test_a_failing_dependent_preserves_what_was_already_acquired(
        self, gates, bar, tmp_path
    ):
        """
        The whole point of the try/except around the recursion: a broken
        instrument must not cost the points measured before it broke.

        """
        readings = iter([1.0, 2.0])

        def flaky():
            return next(readings)

        dependent = Parameter("signal", unit="A", instrument=gates, get_cmd=flaky)
        swept = Sweep(gates.x, 0.0, 1.0, num=5)
        dataset = make_dataset(swept)
        memory = zarr.MemoryStore()

        with pytest.raises(StopIteration):
            step(
                dataset,
                [swept],
                [dependent],
                bar,
                memory_store=memory,
                disk_store=zarr.DirectoryStore(str(tmp_path / "partial.zarr")),
            )

        preserved = xr.open_zarr(memory, consolidated=False)["signal"].values
        np.testing.assert_allclose(preserved[:2], [1.0, 2.0])

    def test_a_keyboard_interrupt_preserves_the_in_memory_dataset(
        self, gates, bar, tmp_path
    ):
        """
        Ctrl-C is the ordinary way to end a run early, and it is caught
        explicitly because it is not an ``Exception``.

        """
        readings = iter([7.0])

        def interrupted():
            try:
                return next(readings)
            except StopIteration:
                raise KeyboardInterrupt from None

        dependent = Parameter("signal", unit="A", instrument=gates, get_cmd=interrupted)
        swept = Sweep(gates.x, 0.0, 1.0, num=4)
        dataset = make_dataset(swept)
        memory = zarr.MemoryStore()

        with pytest.raises(KeyboardInterrupt):
            step(dataset, [swept], [dependent], bar, memory_store=memory)

        preserved = xr.open_zarr(memory, consolidated=False)["signal"].values
        assert preserved[0] == pytest.approx(7.0)

    def test_a_failed_preservation_is_logged_and_the_original_error_wins(
        self, gates, signal, bar, caplog, monkeypatch
    ):
        """
        If the rescue write itself fails, the caller must still see the error
        that ended the sweep, not the one from the rescue attempt.

        """

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
                memory_store=zarr.MemoryStore(),
            )

        assert "Failed to preserve in-memory state" in caplog.text


class TestSaveInterval:
    """``save_interval`` throttles checkpoints during long sweeps."""

    def test_a_long_interval_still_checkpoints_at_the_end(
        self, gates, signal, bar, tmp_path
    ):
        """
        Intermediate checkpoints are skipped, but the unconditional final one
        means the store on disk is never left behind the dataset.

        """
        swept = Sweep(gates.x, 0.0, 1.0, num=4)
        dataset = make_dataset(swept)
        memory = zarr.MemoryStore()
        disk = zarr.DirectoryStore(str(tmp_path / "throttled.zarr"))
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
    """Calls with no slow sweep at all, and the store combinations."""

    def test_no_sweeps_reads_each_dependent_once(self, gates, signal, bar, tmp_path):
        """
        A zero-dimensional measurement: no coordinates, one reading, one
        point on the bar.

        """
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
            memory_store=zarr.MemoryStore(),
        )

        assert dataset["signal"].item() == pytest.approx(50.0)
        assert bar.n == 1

    def test_a_disk_store_without_a_memory_store_is_written_directly(
        self, gates, signal, bar, tmp_path, caplog
    ):
        """
        The incremental checkpoint needs a memory store to diff against; with
        only a disk store it falls back to appending the dataset itself.

        """
        swept = Sweep(gates.x, 0.0, 1.0, num=3)
        dataset = make_dataset(swept)
        disk = zarr.DirectoryStore(str(tmp_path / "direct.zarr"))

        step(dataset, [swept], [signal], bar, disk_store=disk, verbose=True)

        assert "directly to disk store" in caplog.text
        np.testing.assert_allclose(
            xr.open_zarr(disk, consolidated=False)["signal"].values,
            100.0 * swept.values,
        )
