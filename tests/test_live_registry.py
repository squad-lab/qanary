"""Tests for publishing measurements through the live registry."""

from unittest import TestCase
from unittest.mock import patch

import qimchi_connect.producer
import xarray as xr
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
        store = {}
        dataset.to_zarr(store=store, mode="w", zarr_format=2)

        port = measure._register_memory_store("run-1", store)

        restored = open_live_measurement_sync("run-1", f"ws://localhost:{port}")
        xr.testing.assert_identical(restored, dataset)

    def test_the_snapshot_tracks_writes_made_after_publication(self):
        store = {}
        xr.Dataset({"signal": ("x", [1.0])}, coords={"x": [0]}).to_zarr(
            store=store, mode="w", zarr_format=2
        )

        port = measure._register_memory_store("growing", store)
        endpoint = f"ws://localhost:{port}"
        first = open_live_measurement_sync("growing", endpoint)

        xr.Dataset({"signal": ("x", [1.0, 2.0, 3.0])}, coords={"x": [0, 1, 2]}).to_zarr(
            store=store, mode="w", zarr_format=2
        )
        live_server._clear_snapshot_cache()
        second = open_live_measurement_sync("growing", endpoint)

        assert first["signal"].size == 1
        assert second["signal"].size == 3

    def test_publication_names_qanary_as_the_source(self):
        store = {}
        xr.Dataset({"signal": ("x", [1.0])}, coords={"x": [0]}).to_zarr(
            store=store, mode="w", zarr_format=2
        )

        port = measure._register_memory_store("described", store)

        restored = open_live_measurement_sync("described", f"ws://localhost:{port}")
        assert restored.encoding["qimchi_connect_source"] == {
            "source_package": "qanary",
            "source_format": "zarr",
        }

    def test_registration_maintains_the_registry_once(self):
        store = {}
        xr.Dataset({"signal": ("x", [1.0])}, coords={"x": [0]}).to_zarr(
            store=store, mode="w", zarr_format=2
        )

        measure._register_memory_store("run-2", store)

        self.maintain.assert_called_once_with(retention_days=7)

    def test_unregistering_stops_serving_the_measurement(self):
        store = {}
        xr.Dataset({"signal": ("x", [1.0])}, coords={"x": [0]}).to_zarr(
            store=store, mode="w", zarr_format=2
        )
        measure._register_memory_store("transient", store)

        measure._unregister_memory_store("transient")

        assert live_server._provider_entry("transient") is None
