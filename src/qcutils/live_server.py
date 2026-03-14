"""
WebSocket server for broadcasting live measurement data from memory stores.

This module runs a WebSocket server that allows external processes (like the
FastAPI backend) to read data from in-memory Zarr stores in real-time.

"""

import asyncio
import json
import threading
import numpy as np
import zarr
import xarray as xr
from typing import Optional, Set

import websockets
from websockets.server import WebSocketServerProtocol

# Local imports
from qcutils.logger import get_logger

logger = get_logger(__name__)

# Global WebSocket server state
_WS_SERVER: Optional[asyncio.AbstractServer] = None
_WS_THREAD: Optional[threading.Thread] = None
_WS_LOOP: Optional[asyncio.AbstractEventLoop] = None
_CLIENTS: Set[WebSocketServerProtocol] = set()
_MEMORY_STORES_REF = None
_WS_PORT: int = 0  # Store the actual port number


def _sanitize_for_json(value):
    """
    Recursively convert numpy values to JSON-serializable Python types.

    """
    if isinstance(value, np.ndarray):
        return [_sanitize_for_json(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_for_json(item) for item in value]
    return value


def set_memory_stores_reference(stores_dict):
    """Set reference to the LIVE_MEMORY_STORES dictionary."""
    global _MEMORY_STORES_REF
    _MEMORY_STORES_REF = stores_dict


async def _handle_client(websocket: WebSocketServerProtocol):
    """Handle incoming WebSocket client connections."""
    _CLIENTS.add(websocket)
    logger.debug(f"WebSocket client connected from {websocket.remote_address}")

    try:
        async for message in websocket:
            try:
                request = json.loads(message)
                response = await _process_request(request)
                await websocket.send(json.dumps(_sanitize_for_json(response)))
            except json.JSONDecodeError:
                await websocket.send(
                    json.dumps({"error": "Invalid JSON", "success": False})
                )
            except websockets.ConnectionClosedOK:
                # TODO: Monitor for any issues
                pass
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                await websocket.send(json.dumps({"error": str(e), "success": False}))
    except websockets.exceptions.ConnectionClosed:
        pass
        logger.debug(f"WebSocket client disconnected from {websocket.remote_address}")
    finally:
        _CLIENTS.discard(websocket)


async def _process_request(request: dict) -> dict:
    """Process a data request from a client."""
    action = request.get("action")
    measurement_id = request.get("measurement_id")

    if action == "list":
        # List all available measurements
        if _MEMORY_STORES_REF is None:
            return {"success": False, "error": "No memory stores available"}

        measurements = []
        for mid, entry in _MEMORY_STORES_REF.items():
            store = entry.store if hasattr(entry, "store") else entry
            if store is not None:
                measurements.append(mid)

        return {"success": True, "measurements": measurements}

    elif action == "get_data":
        # Get data from a specific measurement
        if not measurement_id:
            return {"success": False, "error": "measurement_id required"}

        if _MEMORY_STORES_REF is None:
            return {"success": False, "error": "No memory stores available"}

        entry = _MEMORY_STORES_REF.get(measurement_id)
        if entry is None:
            return {
                "success": False,
                "error": f"Measurement {measurement_id} not found",
            }

        store = entry.store if hasattr(entry, "store") else entry
        if store is None:
            return {
                "success": False,
                "error": f"Store not available for {measurement_id}",
            }

        try:
            # Open dataset from memory store
            # NOTE: `consolidated=False` because in-memory stores are not always consolidated
            ds = xr.open_zarr(store=store, consolidated=False)

            # Get requested variables
            variables = request.get("variables", [])
            if not variables:
                # Return all variable names if none specified
                return {
                    "success": True,
                    "measurement_id": measurement_id,
                    "coords": list(ds.coords.keys()),
                    "data_vars": list(ds.data_vars.keys()),
                    "attrs": dict(ds.attrs),
                }

            # Return data for requested variables
            data = {}
            for var in variables:
                if var in ds.coords:
                    data[var] = ds.coords[var].values.tolist()
                elif var in ds.data_vars:
                    data[var] = ds.data_vars[var].values.tolist()

            return {
                "success": True,
                "measurement_id": measurement_id,
                "data": data,
                "attrs": dict(ds.attrs),
            }

        except Exception as e:
            logger.error(f"Error reading data from {measurement_id}: {e}")
            return {"success": False, "error": str(e)}

    elif action == "get_zarr_array":
        # Get raw zarr array data for plotting
        if not measurement_id:
            return {"success": False, "error": "measurement_id required"}

        array_path = request.get("array_path", "")  # e.g., "dmm_v1"

        if _MEMORY_STORES_REF is None:
            return {"success": False, "error": "No memory stores available"}

        entry = _MEMORY_STORES_REF.get(measurement_id)
        if entry is None:
            return {
                "success": False,
                "error": f"Measurement {measurement_id} not found",
            }

        store = entry.store if hasattr(entry, "store") else entry
        if store is None:
            return {
                "success": False,
                "error": f"Store not available for {measurement_id}",
            }

        try:
            # Access zarr array directly
            root = zarr.open_group(store=store, mode="r")
            array = root[array_path] if array_path else root

            return {
                "success": True,
                "measurement_id": measurement_id,
                "array_path": array_path,
                "data": array[:].tolist(),
                "shape": array.shape,
                "dtype": str(array.dtype),
            }

        except Exception as e:
            logger.error(
                f"Error reading zarr array {array_path} from {measurement_id}: {e}"
            )
            return {"success": False, "error": str(e)}

    else:
        return {"success": False, "error": f"Unknown action: {action}"}


def _run_server(host: str, port: int):
    """Run the WebSocket server in a separate thread."""
    global _WS_SERVER, _WS_LOOP, _WS_PORT

    _WS_LOOP = asyncio.new_event_loop()
    asyncio.set_event_loop(_WS_LOOP)

    async def start_server():
        global _WS_SERVER, _WS_PORT
        # Use a 100 MB limit to handle large datasets (default is 1 MB).
        _WS_SERVER = await websockets.serve(
            _handle_client, host, port, max_size=100 * 1024 * 1024
        )
        _WS_PORT = port
        logger.debug(f"WebSocket server started on ws://{host}:{port}")
        await _WS_SERVER.wait_closed()

    try:
        _WS_LOOP.run_until_complete(start_server())
    except Exception as e:
        logger.error(f"WebSocket server error: {e}")
    finally:
        _WS_LOOP.close()


def start_live_server(host: str = "localhost", port: int = 8765) -> bool:
    """
    Start the WebSocket server for live data streaming.

    Args:
        host: Host to bind to (default: localhost)
        port: Port to bind to (default: 8765)

    Returns:
        bool: True if server started successfully, False otherwise
    """
    global _WS_THREAD

    if _WS_THREAD is not None and _WS_THREAD.is_alive():
        logger.warning("WebSocket server already running")
        return True

    _WS_THREAD = threading.Thread(
        target=_run_server, args=(host, port), daemon=True, name="qcutils-live-server"
    )
    _WS_THREAD.start()

    # Give server a moment to start
    import time

    time.sleep(0.5)

    return True


def stop_live_server():
    """Stop the WebSocket server."""
    global _WS_SERVER, _WS_LOOP, _WS_THREAD

    if _WS_SERVER is not None and _WS_LOOP is not None:
        _WS_LOOP.call_soon_threadsafe(_WS_SERVER.close)
        _WS_THREAD = None
        logger.debug("WebSocket server stopped")


def is_server_running() -> bool:
    """Check if the WebSocket server is running."""
    return _WS_THREAD is not None and _WS_THREAD.is_alive()


def get_server_port() -> int:
    """Get the port number the WebSocket server is running on."""
    return _WS_PORT
