"""
Tests for qanary integration with the live registry.

Qanary publishes a measurement by handing Qimchi Connect a snapshot callback
that reads its in-memory Zarr store. These tests verify that a Qimchi client
receives the published data and that the discovery row follows the measurement
through its lifecycle.

"""

from unittest import TestCase
from unittest.mock import patch

import qimchi_connect.producer
import xarray as xr
import zarr
from qimchi_connect import registry as live_db
from qimchi_connect import server as live_server
from qimchi_connect.client import open_live_measurement_sync

from qanary import measure


class LivePublicationTests(TestCase):
    def setUp(self):
        self._database = patch.object(live_db, "maintain_registry")
        maintain = self._database.start()
        maintain.return_value = live_db.RegistryMaintenanceResult((), 0)
        self.maintain = maintain

    def tearDown(self):
        self._database.stop()
        for measurement_id in list(qimchi_connect.producer._REGISTRATIONS):
            measure._unregister_memory_store(measurement_id)
        live_server.stop_live_server()

    def test_a_published_store_is_served_over_the_socket(self):
        dataset = xr.Dataset({"signal": ("x", [1.0, 2.0])}, coords={"x": [0, 1]})
        store = zarr.MemoryStore()
        dataset.to_zarr(store=store, mode="w")

        port = measure._register_memory_store("run-1", store)

        restored = open_live_measurement_sync("run-1", f"ws://localhost:{port}")
        xr.testing.assert_identical(restored, dataset)

    def test_the_snapshot_tracks_writes_made_after_publication(self):
        """
        The callback re-reads the store on every request, so a sweep that keeps
        writing is visible without re-registering.

        """
        store = zarr.MemoryStore()
        xr.Dataset({"signal": ("x", [1.0])}, coords={"x": [0]}).to_zarr(
            store=store, mode="w"
        )

        port = measure._register_memory_store("growing", store)
        endpoint = f"ws://localhost:{port}"
        first = open_live_measurement_sync("growing", endpoint)

        xr.Dataset({"signal": ("x", [1.0, 2.0, 3.0])}, coords={"x": [0, 1, 2]}).to_zarr(
            store=store, mode="w"
        )
        live_server._clear_snapshot_cache()
        second = open_live_measurement_sync("growing", endpoint)

        assert first["signal"].size == 1
        assert second["signal"].size == 3

    def test_publication_names_qanary_as_the_source(self):
        store = zarr.MemoryStore()
        xr.Dataset({"signal": ("x", [1.0])}, coords={"x": [0]}).to_zarr(
            store=store, mode="w"
        )

        port = measure._register_memory_store("described", store)

        restored = open_live_measurement_sync("described", f"ws://localhost:{port}")
        assert restored.encoding["qimchi_connect_source"] == {
            "source_package": "qanary",
            "source_format": "zarr",
        }

    def test_registration_maintains_the_registry_once(self):
        store = zarr.MemoryStore()
        xr.Dataset({"signal": ("x", [1.0])}, coords={"x": [0]}).to_zarr(
            store=store, mode="w"
        )

        measure._register_memory_store("run-2", store)

        self.maintain.assert_called_once_with(retention_days=7)

    def test_unregistering_stops_serving_the_measurement(self):
        store = zarr.MemoryStore()
        xr.Dataset({"signal": ("x", [1.0])}, coords={"x": [0]}).to_zarr(
            store=store, mode="w"
        )
        measure._register_memory_store("transient", store)

        measure._unregister_memory_store("transient")

        assert live_server._provider_entry("transient") is None
