"""
Shared fixtures for the qcutils test suite.

Nothing here talks to hardware. Where a test needs a real QCoDeS
``Parameter`` -- one with a unit, a label and a root instrument -- it gets a
``DummyInstrument``; where it only needs something to call, it gets a small
fake defined in the test module itself.

Process-wide state has to be contained, or tests leak into each other:

- QCoDeS keeps a process-wide instrument registry keyed by name, so an
  instrument left open makes the next test asking for that name fail with a
  duplicate-name error. ``instrument`` closes everything it created.
- The live registry defaults to ``~/.qcutils/live_measurements.db``. Tests
  that register a measurement point it at a temporary file instead, so a run
  of the suite never touches the developer's own database.

"""

from itertools import count
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import qimchi_connect.producer
from qcodes.instrument_drivers.mock_instruments import (
    DummyChannelInstrument,
    DummyInstrument,
)
from qcodes.parameters import Parameter
from qimchi_connect import close_live_measurement
from qimchi_connect import registry as live_db
from qimchi_connect import server as live_server

from qcutils import sweep as sweep_module
from qcutils.measure import Measurement, Station

_NAMES = count()


@pytest.fixture
def instrument():
    """
    Factory for throwaway QCoDeS instruments.

    Yields:
        Callable[..., DummyInstrument]: Called as ``instrument("x", "y")`` to
            build a gated instrument, or ``instrument(channels=True)`` for a
            channelled one. Names are unique per call.

    """
    created = []

    def _make(*gates: str, channels: bool = False):
        name = f"qcutils_test_{next(_NAMES)}"
        if channels:
            inst = DummyChannelInstrument(name)
        else:
            inst = DummyInstrument(name, gates=list(gates) or ["ch1"])
        created.append(inst)
        return inst

    yield _make

    for inst in created:
        inst.close()


@pytest.fixture
def gates(instrument):
    """
    A two-gate instrument, both gates parked at zero.

    Returns:
        DummyInstrument: Instrument exposing ``x`` and ``y``.

    """
    inst = instrument("x", "y")
    inst.x(0.0)
    inst.y(0.0)
    return inst


@pytest.fixture
def signal(gates):
    """
    A dependent encoding the gate positions it was read at.

    Returns:
        Parameter: Reads ``100 * x + y``.

    """
    return Parameter(
        "signal",
        unit="A",
        label="Signal",
        instrument=gates,
        get_cmd=lambda: 100.0 * gates.x() + gates.y(),
    )


@pytest.fixture
def station(gates):
    """
    A station holding the dummy instrument and one of its gates.

    Returns:
        Station: Ready to hand to a Measurement.

    """
    built = Station("test-station")
    built.instruments.append(gates)
    built.add_parameter("x_gate", "X gate", gates.x)
    return built


@pytest.fixture
def measurement(tmp_path, station, live_registry, offline_live_server):
    """
    A measurement rooted in a temporary directory, with the slow and
    external parts stubbed.

    Yields:
        Measurement: Ready to run.

    """
    with patch.object(
        Measurement, "get_installed_packages", return_value="qcutils==0.8.0\n"
    ):
        yield Measurement(
            wafer_id="W1",
            device_type="hall-bar",
            sample_name="S1",
            experiment_name="pinchoff",
            station=station,
            data_location=str(tmp_path / "data"),
            metadata={"operator": "test"},
            fridge_name="triton",
        )


@pytest.fixture(autouse=True)
def isolated_qcutils_home(tmp_path, monkeypatch):
    """
    Keep every test out of the real ``~/.qcutils``.

    QCUtils keeps its temporary live Zarr stores there. Qimchi Connect's
    discovery registry is isolated separately by the ``live_registry`` fixture.

    """
    monkeypatch.setenv("QCUTILS_HOME", str(tmp_path / "qcutils-home"))


@pytest.fixture
def live_registry(tmp_path):
    """
    Point the live registry at a temporary database.

    Yields:
        Path: The database file the registry writes to for this test.

    """
    database = tmp_path / "live_measurements.db"
    live_db.configure_database(database)
    live_db.init_database()
    yield database
    live_db.configure_database(None)


@pytest.fixture
def offline_live_server():
    """
    Pretend the WebSocket server is already up.

    Publishing a measurement otherwise binds a port and starts a server
    thread, which is Qimchi Connect's responsibility to test. Only the
    socket work is faked: the provider is really registered and the discovery
    row really written, so tests can assert on what was published.

    Yields:
        int: The port number the patched server reports.

    """
    port = 9999
    with (
        patch.object(live_server, "is_server_running", return_value=True),
        patch.object(live_server, "get_server_port", return_value=port),
        patch.object(live_server, "get_server_host", return_value="localhost"),
    ):
        yield port

    for measurement_id in list(qimchi_connect.producer._REGISTRATIONS):
        close_live_measurement(measurement_id)
        live_server.unregister_snapshot_provider(measurement_id)


@pytest.fixture
def quiet_registry_maintenance():
    """
    Stub the registry maintenance sweep that runs on every registration.

    Yields:
        MagicMock: The patched ``maintain_registry``.

    """
    with patch.object(live_db, "maintain_registry") as maintain:
        maintain.return_value = SimpleNamespace(
            stale_measurement_ids=(),
            deleted_count=0,
        )
        yield maintain


@pytest.fixture(autouse=True)
def reset_sweep_globals():
    """
    Clear the module-level state that survives between sweeps.

    ``last_save`` gates checkpointing and the digest cache records what a
    checkpoint believes is already on disk. Both are module globals, so a test
    that leaves them set changes what the next test's first checkpoint does.

    """
    sweep_module.last_save = 0
    sweep_module.reset_disk_persist_cache()
    yield
    sweep_module.last_save = 0
    sweep_module.reset_disk_persist_cache()


class RecordingBar:
    """
    Stand-in for the tqdm bar, recording every increment it is handed.

    Attributes:
        n (int): Total points advanced so far.
        increments (list[int]): Every increment, in order.

    """

    def __init__(self) -> None:
        self.n = 0
        self.increments: list[int] = []

    def update(self, points: int) -> None:
        """
        Advance the bar.

        Args:
            points (int): Number of points completed.

        """
        self.increments.append(points)
        self.n += points


@pytest.fixture
def bar():
    """
    A progress bar that records rather than draws.

    Returns:
        RecordingBar: Fresh bar.

    """
    return RecordingBar()
