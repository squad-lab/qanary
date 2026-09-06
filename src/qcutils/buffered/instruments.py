"""
Buffered sweep nodes for specific instruments.

.. deprecated::
    These instrument-specific node classes are moving to the ``drivers``
    package. They remain importable from here for now, but no new node
    classes should be added here.

    Nothing inside QCUtils imports this module. It exists for measurement
    scripts, which can change their imports once ``drivers`` provides these
    classes.

"""

import warnings
from time import sleep, time
from typing import Sequence

import numpy as np
from pyvisa.constants import StatusCode
from pyvisa.errors import VisaIOError
from qcodes.instrument import Instrument
from qcodes.parameters import Parameter

from qcutils.sweep import Sweep

warnings.warn(
    "qcutils.buffered.instruments is deprecated and will move to the `drivers` "
    "package. Import the node classes from there once it provides them.",
    DeprecationWarning,
    stacklevel=2,
)

# Public API: the node classes a buffered sweep tree is built from.
__all__ = [
    "BufferedNodeBase",
    "NodeMFLI",
    "NodeUHFLI",
    "NodeKeysightDMM",
    "NodeQDAC2",
    "NodeKeysightVNA",
    "NodeBaselDAC",
    "NodeDelay",
]


class BufferedNodeBase:
    def __init__(self, inst: Instrument) -> None:
        """
        Initialize a buffered-tree node around an instrument.

        A node may drive sweeps, acquire dependents, or do both. Concrete node
        classes implement the operations their hardware supports.

        Args:
            inst (Instrument): QCoDeS instrument controlled by this node.

        """
        self.buffered = True
        self.toplevel = False
        self._processed_sweeps = False
        self._processed_dependents = False
        self.core = inst

        self.endnode = True
        self.sweepnode = True

    def _process_sweeps(self, sweep: Sweep | Sequence[Sweep]) -> None:
        """
        Normalize and validate the sweeps assigned to this node.

        Args:
            sweep (Sweep | Sequence[Sweep]): One or two sweeps belonging to the
                same instrument and using the same delay.

        Raises:
            AssertionError: If the sweeps use different instruments or delays,
                or more than two dimensions are supplied.

        """
        if not isinstance(sweep, Sequence):
            self.sweeps = [sweep]
        else:
            self.sweeps = sweep

        instruments = [
            param.underlying_instrument for sw in self.sweeps for param in sw.parameter
        ]
        assert len(set(instruments)) == 1, (
            "All sweeps of the buffered node must be from the same instrument"
        )

        delays = [sw.delay for sw in self.sweeps]
        assert len(set(delays)) == 1, (
            "All sweeps of the buffered node must have the same delay"
        )

        assert len(self.sweeps) <= 2, "Maximum 2D sweep supported"

        self.delay = delays[0]
        self.num = 1
        for sw in self.sweeps:
            self.num *= sw.num
        self.num_tup = [sw.num for sw in self.sweeps]
        self.start_delay = [sw.start_delay for sw in self.sweeps]
        self.endnode = False
        self.dims = len(self.sweeps)

    def _process_dependents(self, dependent: Parameter | Sequence[Parameter]) -> None:
        """
        Normalize one or more dependent parameters to a sequence.

        Args:
            dependent (Parameter | Sequence[Parameter]): Parameters acquired by
                this node.

        """
        if not isinstance(dependent, Sequence):
            self.dependents = [dependent]
        else:
            self.dependents = dependent

    def abort(self) -> None:
        """Provide a no-op cleanup hook for nodes without active resources."""


class NodeMFLI(BufferedNodeBase):
    """
    Zurich Instruments MFLI acquisition node using the LabOne DAQ module.

    The node uses hardware-trigger mode. For two-dimensional geometry, rows are
    trigger events and columns are points recorded per trigger. The upstream
    sweep node must provide a trigger pulse at least 100 microseconds wide.

    """

    def __init__(self, inst: Instrument, *args, **kwargs) -> None:
        super().__init__(inst=inst, *args, **kwargs)

        self.core = self.core.core
        self.serial = self.core.serial
        self.daq = self.core.session.daq_server

        self.daq_module = self.daq.dataAcquisitionModule()
        self.daq_module.set("device", self.serial)
        self.daq_module.set("type", 6)

        self._subs: list[str] = []

    def register_dependent(
        self,
        dependent: Parameter | Sequence[Parameter],
        num: int | Sequence[int],
        delay: float,
        input_trigger: int = 1,
        *,
        force_trigger: bool = False,
        trigger_delay: float = 0.0,
        trigger_level: float = 0.5,
        tc_factor: float = 1.0,
        grid_mode: str = "linear",
        edge: str = "rising",
        endless: bool = False,
        count: int = 1,
    ) -> None:
        """
        Configure and start a hardware-triggered DAQ acquisition.

        Args:
            dependent (Parameter | Sequence[Parameter]): Parameters exposing a
                Zurich Instruments ``zi_node`` path.
            num (int | Sequence[int]): Point count, or ``(rows, columns)`` for a
                two-dimensional acquisition.
            delay (float): Innermost step time in seconds.
            input_trigger (int): Trigger input number. Defaults to 1.
            force_trigger (bool): Force a trigger after arming. Defaults to
                False.
            trigger_delay (float): Additional delay before the first sample.
                Defaults to 0.
            trigger_level (float): Trigger threshold. Defaults to 0.5.
            tc_factor (float): Set the demodulator time constant to
                ``delay / tc_factor``. Defaults to 1.
            grid_mode (str): ``"nearest"``, ``"linear"``, or ``"exact"``.
                Defaults to ``"linear"``.
            edge (str): ``"rising"``, ``"falling"``, or ``"both"``. Defaults
                to ``"rising"``.
            endless (bool): Enable continuous acquisition. Defaults to False.
            count (int): Grids to acquire when *endless* is false. Defaults to
                1.

        Raises:
            ValueError: If *grid_mode* is unsupported.

        """
        # Normalize the dependent list.
        if isinstance(dependent, Sequence):
            self.dependents = [dep.zi_node.lower() for dep in dependent]
        else:
            self.dependents = [dependent.zi_node.lower()]

        if grid_mode not in ["nearest", "linear", "exact"]:
            raise ValueError(
                f"Invalid grid_mode: {grid_mode}. Must be nearest, linear or exact."
            )

        # Save whether the module should be triggered after arming.
        self._force_trigger = force_trigger

        # Configure how samples are aligned to the requested grid.
        self.daq_module.set("grid/mode", grid_mode)

        self.daq.setInt(f"/{self.serial}/demods/0/enable", 1)
        self.daq.setDouble(
            f"/{self.serial}/demods/0/timeconstant", float(delay) / tc_factor
        )

        self.daq_module.finish()
        self.daq_module.unsubscribe("*")
        self.daq_module.set("clearhistory", 1)

        self.daq_module.set(
            "triggernode", f"/{self.serial}/demods/0/sample.TrigIn{int(input_trigger)}"
        )

        # Configure the selected trigger input.
        self.daq.setDouble(
            f"/{self.serial}/triggers/in/{int(input_trigger) - 1}/level",
            trigger_level,
        )

        edge_map = {"rising": 1, "falling": 2, "both": 3}
        self.daq_module.set("edge", edge_map.get(edge, 1))

        self.daq_module.set("endless", 1 if endless else 0)
        if not endless:
            self.daq_module.set("count", int(count))

        if isinstance(num, Sequence) and not isinstance(num, (str, bytes)):
            rows, cols = int(num[0]), int(num[1])
        else:
            rows, cols = 1, int(num)

        self.daq_module.set("grid/rows", rows)
        self.daq_module.set("grid/cols", cols)

        if grid_mode != "exact":
            duration = float(delay) * cols
            self.daq_module.set("duration", duration)

        # Sample near the end of each ramp step, adjusted by trigger_delay.
        self.daq_module.set("delay", float(delay) + trigger_delay)

        self.daq_module.set("holdoff/time", max(0.0, float(delay) * (cols - 0.5)))

        self._subs = []
        for dep in self.dependents:
            path = f"/{self.serial}{dep}"
            self.daq_module.subscribe(path)
            self._subs.append(path)

        self.daq_module.execute()

        if self._force_trigger:
            sleep(0.2)  # Allow the DAQ module to finish arming.
            self.daq_module.set("forcetrigger", 1)

    def fetch(self, *, timeout: float = 15.0) -> list[np.ndarray]:
        """
        Wait for DAQ completion and read subscribed values.

        Args:
            timeout (float): Maximum wait before reading available data.
                Defaults to 15 seconds.

        Returns:
            list[np.ndarray]: One flattened array per dependent.

        """
        t0 = time()
        while not self.daq_module.finished():
            sleep(0.05)
            if time() - t0 > timeout:
                break

        result = self.daq_module.read()
        self.daq_module.finish()

        arrays: list[np.ndarray] = []
        for dep in self.dependents:
            dep_split = dep.split("/")
            data = result[self.serial][dep_split[1]][dep_split[2]][dep_split[3]][0][
                "value"
            ]
            arrays.append(np.array(data).flatten())

        return arrays


class NodeUHFLI(BufferedNodeBase):
    """
    Zurich Instruments UHFLI buffered sweep and acquisition node.

    Dependents may use the hardware-triggered DAQ module or the LabOne Sweeper
    module. The upstream sweep node must provide a trigger pulse at least
    100 microseconds wide for DAQ acquisition.

    """

    _shared = {}

    def __init__(self, inst: Instrument, *args, **kwargs):
        super().__init__(inst=inst, *args, **kwargs)

        self.core = self.core.core
        self.serial = self.core.serial
        self.daq = self.core.session.daq_server

        self.daq_module = self.daq.dataAcquisitionModule()
        self.daq_module.set("device", self.serial)
        self.daq_module.set("type", 6)

        # Share one Sweeper module and its subscriptions per physical UHFLI.
        if self.serial not in self._shared:
            sweeper = self.daq.sweep()
            sweeper.set("device", self.serial)

            self._shared[self.serial] = {
                "sweeper_module": sweeper,
                "subs": [],
                "sweeper_fields": [],
            }

        self._ctx = self._shared[self.serial]

        self.sweeper_module = self._ctx["sweeper_module"]

        self._active_acquisition = None

    def register_sweep(
        self,
        sweep: Sequence[Sweep],
        input_trigger: int | None = None,
        output_trigger: int | None = None,
        trigger_type: str | None = None,
        trigger_width: float | None = None,
        *,
        spacing: str = "lin",
        averaging: int = 1,
        averaging_tc: int = 5,
        sweep_order: int = 3,
        settling_inaccuracy: float = 100 * 1e-6,
        phase_unwrap: bool = False,
        **kwargs,
    ) -> tuple[None, int, None]:
        """
        Configure a one-dimensional LabOne Sweeper Module sweep.

        Args:
            sweep (Sequence[Sweep]): Sequence containing exactly one sweep of a
                parameter with a ``zi_node`` attribute.
            input_trigger (int | None): Accepted for the common node interface
                and ignored by this acquisition mode.
            output_trigger (int | None): Accepted for the common node interface
                and ignored by this acquisition mode.
            trigger_type (str | None): Accepted for the common node interface
                and ignored by this acquisition mode.
            trigger_width (float | None): Accepted for the common node interface
                and ignored by this acquisition mode.
            spacing (str): ``"lin"`` or ``"log"``. Defaults to ``"lin"``.
            averaging (int): Samples averaged per sweep point. Defaults to 1.
            averaging_tc (int): Time constants allowed per point. Defaults to
                5.
            sweep_order (int): Demodulator filter order. Defaults to 3.
            settling_inaccuracy (float): Target filter-settling inaccuracy.
            phase_unwrap (bool): Unwrap phase across 2-pi boundaries. Defaults
                to False.
            **kwargs: Ignored compatibility options.

        Returns:
            tuple[None, int, None]: No trigger type, point count, and no fixed
                step time.

        Raises:
            ValueError: If more than one sweep is supplied, the parameter has
                no ``zi_node``, or *spacing* is unsupported.

        """

        if len(sweep) > 1:
            raise ValueError(
                "Only 1D sweeps are supported with the Sweeper module, with one parameter being swept."
            )
        else:
            single_sweep = sweep[0]

        self.num = int(single_sweep.num)
        self.delay = float(single_sweep.delay)

        sweep_parameter = single_sweep.parameter[0]

        if not hasattr(sweep_parameter, "zi_node"):
            raise ValueError(
                f"Parameter {sweep_parameter.full_name!r} cannot be swept "
                "with the Zurich Instruments Sweeper because it has no zi_node."
            )

        gridnode = f"/{self.serial}/{sweep_parameter.zi_node.lower().lstrip('/')}"
        self.sweeper_module.set("gridnode", gridnode)

        self.sweeper_module.set("start", float(single_sweep.start))
        self.sweeper_module.set("stop", float(single_sweep.stop))
        self.sweeper_module.set("samplecount", int(self.num))

        if spacing == "lin":
            xmapping = 0
        elif spacing == "log":
            xmapping = 1
        else:
            raise ValueError(f"Invalid spacing: {spacing}. Must be 'lin' or 'log'.")

        self.sweeper_module.set("scan", 0)  # Sequential forward scan.
        self.sweeper_module.set("xmapping", xmapping)

        self.sweeper_module.set("bandwidthcontrol", 2)  # Automatic selection.
        self.sweeper_module.set("bandwidthoverlap", 0)
        self.sweeper_module.set("loopcount", 1)

        self.sweeper_module.set("settling/time", 0)

        self.sweeper_module.set(
            "settling/inaccuracy",
            float(settling_inaccuracy),
        )

        self.sweeper_module.set(
            "averaging/sample",
            int(averaging),
        )

        self.sweeper_module.set(
            "averaging/tc",
            int(averaging_tc),
        )

        self.sweeper_module.set(
            "order",
            int(sweep_order),
        )

        self.sweeper_module.set(
            "phaseunwrap",
            int(phase_unwrap),
        )

        num_points = self.num

        return None, num_points, None

    def run_sweep(self):
        """Start the configured LabOne sweep."""
        self.daq_module.finish()
        self.sweeper_module.execute()
        if self.toplevel:
            sleep((self.num + 1) * self.delay)

    def _register_daq_dependent(
        self,
        dependent: Parameter | Sequence[Parameter],
        num: int | Sequence[int],
        delay: float,
        input_trigger: int = 1,
        *,
        demod_channels: int | Sequence[int] = 0,
        force_trigger: bool = False,
        trigger_delay: float = 0.0,
        trigger_level: float = 0.5,
        tc_factor: float = 1.0,
        grid_mode: str = "linear",
        edge: str = "rising",
        endless: bool = False,
        count: int = 1,
        **kwargs,
    ) -> None:
        """
        Configure and start a hardware-triggered DAQ acquisition.

        Args:
            dependent (Parameter | Sequence[Parameter]): Parameters exposing a
                Zurich Instruments ``zi_node`` path.
            num (int | Sequence[int]): Point count, or ``(rows, columns)`` for a
                two-dimensional acquisition.
            delay (float): Innermost step time in seconds.
            input_trigger (int): Trigger input number. Defaults to 1.
            demod_channels (int | Sequence[int]): Demodulators to enable.
                Defaults to 0.
            force_trigger (bool): Force a trigger after arming. Defaults to
                False.
            trigger_delay (float): Additional delay before the first sample.
                Defaults to 0.
            trigger_level (float): Trigger threshold. Defaults to 0.5.
            tc_factor (float): Set each demodulator time constant to
                ``delay / tc_factor``. Defaults to 1.
            grid_mode (str): ``"nearest"``, ``"linear"``, or ``"exact"``.
                Defaults to ``"linear"``.
            edge (str): ``"rising"``, ``"falling"``, or ``"both"``. Defaults
                to ``"rising"``.
            endless (bool): Enable continuous acquisition. Defaults to False.
            count (int): Grids to acquire when *endless* is false. Defaults to
                1.
            **kwargs: Ignored compatibility options.

        Raises:
            ValueError: If *grid_mode* is unsupported.

        """
        # Normalize the dependent list.
        if isinstance(dependent, Sequence):
            self.dependents = [dep.zi_node.lower() for dep in dependent]
        else:
            self.dependents = [dependent.zi_node.lower()]

        if grid_mode not in ["nearest", "linear", "exact"]:
            raise ValueError(
                f"Invalid grid_mode: {grid_mode}. Must be nearest, linear or exact."
            )

        # Save whether the module should be triggered after arming.
        self._force_trigger = force_trigger

        # Configure how samples are aligned to the requested grid.
        self.daq_module.set("grid/mode", grid_mode)

        for demod in np.atleast_1d(demod_channels):
            self.daq.setInt(f"/{self.serial}/demods/{demod}/enable", 1)
            self.daq.setDouble(
                f"/{self.serial}/demods/{demod}/timeconstant", float(delay) / tc_factor
            )

        self.daq_module.finish()
        self.daq_module.unsubscribe("*")
        self.daq_module.set("clearhistory", 1)

        for demod in np.atleast_1d(demod_channels):
            self.daq_module.set(
                "triggernode",
                f"/{self.serial}/demods/{demod}/sample.TrigIn{int(input_trigger)}",
            )

        # Configure the selected trigger input.
        self.daq.setDouble(
            f"/{self.serial}/triggers/in/{int(input_trigger) - 1}/level",
            trigger_level,
        )

        # Use high-impedance mode for every trigger input.
        for i in [0, 1, 2, 3]:
            self.daq.set(f"/{self.serial}/triggers/in/{i}/imp50", 0)

        edge_map = {"rising": 1, "falling": 2, "both": 3}
        self.daq_module.set("edge", edge_map.get(edge, 1))

        self.daq_module.set("endless", 1 if endless else 0)
        if not endless:
            self.daq_module.set("count", int(count))

        if isinstance(num, Sequence) and not isinstance(num, (str, bytes)):
            rows, cols = int(num[0]), int(num[1])
        else:
            rows, cols = 1, int(num)

        self.daq_module.set("grid/rows", rows)
        self.daq_module.set("grid/cols", cols)

        if grid_mode != "exact":
            duration = float(delay) * cols
            self.daq_module.set("duration", duration)

        # Sample near the end of each ramp step, adjusted by trigger_delay.
        self.daq_module.set("delay", float(delay) + trigger_delay)

        self.daq_module.set("holdoff/time", max(0.0, float(delay) * (cols - 0.5)))

        self._subs = []
        for dep in self.dependents:
            path = f"/{self.serial}{dep}"
            self.daq_module.subscribe(path)
            self._subs.append(path)

        self.daq_module.execute()

        if self._force_trigger:
            sleep(0.2)  # Allow the DAQ module to finish arming.
            self.daq_module.set("forcetrigger", 1)

    def _register_sweeper_dependent(
        self,
        dependent: Parameter | Sequence[Parameter],
        num: int | Sequence[int],
        delay: float,
        input_trigger: int | None = None,
        **kwargs,
    ) -> None:
        """
        Subscribe dependents to the configured LabOne Sweeper Module.

        Args:
            dependent (Parameter | Sequence[Parameter]): Parameters exposing a
                supported ``zi_node`` sample field.
            num (int | Sequence[int]): Accepted for the common node interface.
            delay (float): Accepted for the common node interface.
            input_trigger (int | None): Accepted for the common node interface.
            **kwargs: Ignored compatibility options.

        Raises:
            ValueError: If this is not an acquisition-only end node or a
                dependent uses an unsupported sample field.

        """
        if not self.endnode:
            raise ValueError("Sweeper dependents can only be registered on end nodes.")

        if isinstance(dependent, Sequence):
            dependents = list(dependent)
        else:
            dependents = [dependent]

        self.dependents = dependents

        # Map each QCoDeS dependent to its Sweeper sample field.
        sweeper_fields = []

        # Subscribe to each demodulator sample node only once.
        sample_paths = set()

        for dep in dependents:
            zi_node = dep.zi_node.lower()

            # For example, /demods/0/sample.r maps to /demods/0/sample.
            if zi_node.endswith(".r"):
                sample_path = zi_node.removesuffix(".r")
                field = "r"

            elif zi_node.endswith(".theta"):
                sample_path = zi_node.removesuffix(".theta")
                field = "phase"

            elif zi_node.endswith(".x"):
                sample_path = zi_node.removesuffix(".x")
                field = "x"

            elif zi_node.endswith(".y"):
                sample_path = zi_node.removesuffix(".y")
                field = "y"

            else:
                raise ValueError(f"Unsupported Sweeper dependent: {zi_node!r}")

            full_path = f"/{self.serial}{sample_path}"

            sample_paths.add(full_path)

            sweeper_fields.append(
                {
                    "dependent": dep,
                    "path": full_path,
                    "field": field,
                }
            )

        self._ctx["subs"] = list(sample_paths)
        self._ctx["sweeper_fields"] = sweeper_fields

        self.sweeper_module.unsubscribe("*")

        for path in self._ctx["subs"]:
            self.sweeper_module.subscribe(path)

    def register_dependent(
        self,
        dependent: Parameter | Sequence[Parameter],
        num: int | list[int],
        delay: float,
        *,
        acquisition: str = "daq",
        **kwargs,
    ) -> None:
        """
        Configure dependents for DAQ or Sweeper acquisition.

        Args:
            dependent (Parameter | Sequence[Parameter]): Parameters exposing a
                Zurich Instruments ``zi_node`` path.
            num (int | list[int]): Point count, or two-dimensional geometry.
            delay (float): Innermost step time in seconds.
            acquisition (str): ``"daq"`` or ``"sweeper"``. Defaults to
                ``"daq"``.
            **kwargs: Mode-specific options forwarded to the selected
                acquisition setup.

        Raises:
            ValueError: If *acquisition* is unsupported or a Sweeper dependent
                is registered on a non-end node.

        """
        if acquisition == "daq":
            self._register_daq_dependent(
                dependent=dependent,
                num=num,
                delay=delay,
                **kwargs,
            )

        elif acquisition == "sweeper":
            self._register_sweeper_dependent(
                dependent=dependent,
                num=num,
                delay=delay,
                **kwargs,
            )

        else:
            raise ValueError(
                f"Unknown acquisition mode {acquisition!r}. "
                "Expected 'daq' or 'sweeper'."
            )

        self._active_acquisition = acquisition

    def fetch(self, *, timeout: float = 15.0) -> list[np.ndarray]:
        """
        Fetch the active DAQ or Sweeper acquisition.

        Args:
            timeout (float): DAQ wait limit in seconds. Sweeper acquisition uses
                twice this value as an allowance beyond its estimated duration.
                Defaults to 15.

        Returns:
            list[np.ndarray]: One flattened array per dependent.

        Raises:
            RuntimeError: If no acquisition has been registered.
            TimeoutError: If a Sweeper acquisition exceeds its estimated time
                plus the timeout allowance.

        """
        if self._active_acquisition == "daq":
            return self._fetch_daq(timeout=timeout)

        if self._active_acquisition == "sweeper":
            return self._fetch_sweeper(timeout=2 * timeout)

        raise RuntimeError("No acquisition has been registered.")

    def _fetch_daq(self, *, timeout: float = 15.0) -> list[np.ndarray]:
        """
        Wait for DAQ completion and read subscribed values.

        Args:
            timeout (float): Maximum wait before reading available data.
                Defaults to 15 seconds.

        Returns:
            list[np.ndarray]: One flattened array per dependent.

        """
        t0 = time()
        while not self.daq_module.finished():
            sleep(0.05)
            if time() - t0 > timeout:
                break

        result = self.daq_module.read()
        self.daq_module.finish()

        arrays: list[np.ndarray] = []
        for dep in self.dependents:
            dep_split = dep.split("/")
            data = result[self.serial][dep_split[1]][dep_split[2]][dep_split[3]][0][
                "value"
            ]
            arrays.append(np.array(data).flatten())

        return arrays

    def _fetch_sweeper(self, *, timeout: float = 30.0) -> list[np.ndarray]:
        """
        Wait for Sweeper completion and read the requested sample fields.

        Args:
            timeout (float): Additional allowance beyond the Sweeper's reported
                remaining time. Defaults to 30 seconds.

        Returns:
            list[np.ndarray]: One flattened array per dependent.

        Raises:
            TimeoutError: If acquisition exceeds its reported remaining time
                plus *timeout*.
            KeyError: If a requested field is absent from the returned sample.

        """
        remaining = self.sweeper_module.getDouble("remainingtime")
        while np.isnan(remaining):
            sleep(0.1)
            remaining = self.sweeper_module.getDouble("remainingtime")

        t0 = time()
        real_timeout = remaining + timeout

        while not self.sweeper_module.finished():
            if time() - t0 > real_timeout:
                self.sweeper_module.finish()
                raise TimeoutError(
                    f"Sweeper acquisition timed out after {real_timeout:.1f} s."
                )

            sleep(0.05)

        self.sweeper_module.finish()
        result = self.sweeper_module.read()

        arrays = []

        for spec in self._ctx["sweeper_fields"]:
            path = spec["path"]
            field = spec["field"]

            # Example path: /dev2793/demods/0/sample
            parts = path.strip("/").split("/")

            device = parts[0]  # dev2793
            demod = parts[2]  # 0

            sample = result[device]["demods"][demod]["sample"][0][0]

            if field not in sample:
                raise KeyError(
                    f"Field {field!r} not found in Sweeper sample. "
                    f"Available fields: {list(sample.keys())}"
                )

            arrays.append(np.asarray(sample[field]).flatten())

        self.sweeper_module.unsubscribe("*")

        return arrays


class NodeKeysightDMM(BufferedNodeBase):
    """
    Keysight 344xxA DMM node using externally triggered reading memory.

    One-dimensional step-triggered acquisition is implemented. A
    two-dimensional point shape is accepted by the shared interface, but its
    DMM count configuration still requires hardware validation. The upstream
    sweep node must provide a trigger pulse at least one millisecond wide.

    """

    def __init__(self, inst: Instrument, *args, **kwargs) -> None:
        super().__init__(inst=inst, *args, **kwargs)
        self.core.trigger.source("IMM")
        self.core.reset()

    def register_dependent(
        self,
        dependent: Parameter | Sequence[Parameter] | None,
        num: int | Sequence[int],
        delay: int | float,
        input_trigger: int = 1,
        trigger_type: str = "step",
    ) -> None:
        """
        Configure a buffered, externally triggered DMM acquisition.

        Args:
            dependent (Parameter | Sequence[Parameter] | None): Parameters
                associated with the DMM readings.
            num (int | Sequence[int]): Point count. A two-dimensional shape is
                interpreted as ``(trigger events, points per trigger)`` but is
                not fully configured yet.
            delay (int | float): Step time used to select trigger delay and
                integration time.
            input_trigger (int): External trigger selector. Only 1 is supported.
            trigger_type (str): Triggering mode. Only ``"step"`` is supported.

        Raises:
            NotImplementedError: If the trigger input, trigger type, or DMM
                model is unsupported.

        """
        self._process_dependents(dependent)
        self.num = num
        self.delay = delay
        self.trigger_type = trigger_type
        if input_trigger == 1:
            self.core.trigger.source("EXT")
        else:
            raise NotImplementedError(
                "Only input_trigger=1 (EXT) is supported for Keysight DMM"
            )
        self.core.trigger.slope("POS")
        self.core.autorange("OFF")
        self.core.autozero("OFF")

        if isinstance(num, Sequence) and not isinstance(num, (str, bytes)):
            # TODO: Configure the 2D counts after validating them on the DMM.
            rows, cols = int(num[0]), int(num[1])  # noqa: F841
        else:
            if self.trigger_type == "step":
                self.core.sample.count(1)
                self.core.trigger.count(self.num)

                if self.core.model == "34410A":
                    self.core.trigger.delay(0.1 * self.delay)
                elif self.core.model == "34461A":
                    self.core.trigger.delay(0.2 * self.delay)
                else:
                    raise NotImplementedError(
                        f"Trigger type 'step' not implemented for {self.core.model}"
                    )

            else:
                raise NotImplementedError(
                    f"Trigger type '{self.trigger_type}' not implemented for Keysight DMM"
                )

        trigger_delay = 0.2 * self.delay
        self.core.trigger.delay(trigger_delay)

        available = 0.8 * (self.delay - trigger_delay)

        for nplc in reversed(self.core.NPLC_list):
            if nplc / self.core.line_frequency() < available:
                self.core.NPLC(nplc)
                break

        self.core.init_measurement()
        sleep(0.1)

    def fetch(self) -> list[np.ndarray]:
        """
        Fetch buffered measurements from the Keysight DMM.

        Returns:
            list[np.ndarray]: List containing a single flattened array of
            measured values.

        Raises:
            VisaIOError: If the instrument read fails or times out.

        """
        if isinstance(self.num, Sequence) and not isinstance(self.num, (str, bytes)):
            num = int(self.num[0]) * int(self.num[1])
        else:
            num = int(self.num)

        try:
            with self.core.timeout.set_to(max(5, num * self.delay + 2)):
                data = self.core.fetch()
        except VisaIOError as exc:
            if exc.error_code != StatusCode.error_timeout:
                raise

            raise
        finally:
            self.abort()

        return [np.asarray(data).flatten()]

    def abort(self) -> None:
        """Abort acquisition and restore settings for ordinary scalar reads."""
        self.core.device_clear()
        self.core.abort_measurement()
        self.core.sample.count(1)
        self.core.trigger.count(1)
        self.core.trigger.source("IMM")


class NodeQDAC2(BufferedNodeBase):
    """QDevil QDAC-II sweep and current-acquisition node."""

    def __init__(
        self,
        inst: Instrument,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize a node around a QDAC-II instrument.

        Args:
            inst (Instrument): QDAC-II QCoDeS instrument.
            *args: Additional arguments passed to :class:`BufferedNodeBase`.
            **kwargs: Additional keyword arguments passed to
                :class:`BufferedNodeBase`.

        """
        super().__init__(inst=inst, *args, **kwargs)
        self.contacts = {}

    def register_sweep(
        self,
        sweep: Sweep | Sequence[Sweep],
        input_trigger: int | None = None,
        output_trigger: int | None = None,
        trigger_type: str = "ramp",
        trigger_width: float = 1e-4,
    ) -> tuple[str, int | list[int], float]:
        """
        Configure a one- or two-dimensional QDAC-II virtual sweep.

        Args:
            sweep (Sweep | Sequence[Sweep]): One or two QDAC-II channel sweeps.
            input_trigger (int | None): Optional external start-trigger number.
            output_trigger (int | None): Optional trigger-output number.
            trigger_type (str): ``"ramp"`` for an output trigger per inner
                ramp or ``"step"`` for one per inner point. One-dimensional
                sweeps support only ``"step"``.
            trigger_width (float): Output-trigger width in seconds. Defaults to
                0.0001.

        Returns:
            tuple[str, int | list[int], float]: Trigger type, point geometry,
                and innermost step time.

        Raises:
            ValueError: If the step delay is shorter than the trigger width or
                *trigger_type* is invalid.
            NotImplementedError: If the requested dimensionality or a
                one-dimensional ramp trigger is unsupported.

        """
        self._process_sweeps(sweep)
        self._trigger_width = trigger_width

        if self.delay < self._trigger_width:
            raise ValueError(
                f"Delay {self.delay} s is less than trigger width {self._trigger_width} s"
            )

        self.core.free_all_triggers()
        if trigger_type not in ["ramp", "step"]:
            raise ValueError('trigger_type must be either "ramp" or "step"')

        if len(self.sweeps) == 2:
            inner_sweep = sweep[1]
            outer_sweep = sweep[0]

            inner_voltages = inner_sweep.values
            outer_voltages = outer_sweep.values

            self.input_trigger = (
                {f"trigin_{input_trigger}": input_trigger} if input_trigger else None
            )
            self.output_trigger = (
                {f"trigout_{output_trigger}": output_trigger}
                if output_trigger
                else None
            )
            self.input_trigger_key = (
                f"trigin_{input_trigger}" if input_trigger else None
            )
            self.output_trigger_key = (
                f"trigout_{output_trigger}" if output_trigger else None
            )

            self.contacts = {
                inner_sweep.parameter[0].name: inner_sweep.parameter[
                    0
                ].instrument._channum,
                outer_sweep.parameter[0].name: outer_sweep.parameter[
                    0
                ].instrument._channum,
            }
            self.arrangement = self.core.arrange(
                contacts=self.contacts,
                output_triggers=self.output_trigger,
            )

            for trig in self.core.external_triggers:
                trig.width_s(self._trigger_width)

            if trigger_type == "ramp":
                num_points = self.num_tup
                self._qdac_sweep = self.arrangement.virtual_sweep2d(
                    inner_contact=inner_sweep.parameter[0].name,
                    outer_contact=outer_sweep.parameter[0].name,
                    inner_voltages=inner_voltages,
                    outer_voltages=outer_voltages,
                    start_sweep_trigger=self.input_trigger_key,
                    inner_step_time_s=self.delay,
                    outer_step_trigger=self.output_trigger_key,
                )
            else:
                num_points = self.num
                self._qdac_sweep = self.arrangement.virtual_sweep2d(
                    inner_contact=inner_sweep.parameter[0].name,
                    outer_contact=outer_sweep.parameter[0].name,
                    inner_voltages=inner_voltages,
                    outer_voltages=outer_voltages,
                    start_sweep_trigger=self.input_trigger_key,
                    inner_step_time_s=self.delay,
                    inner_step_trigger=self.output_trigger_key,
                )

            return trigger_type, num_points, self.delay

        elif len(self.sweeps) == 1:
            num_points = self.num
            sweep = self.sweeps[0]

            self.contacts = {}
            if isinstance(sweep.parameter, Sequence):
                assert len(sweep.parameter) == 1, (
                    "Only one parameter supported for 1D sweeps"
                )
                for param in sweep.parameter:
                    self.contacts[param.name] = param.instrument._channum
            else:
                self.contacts[sweep.parameter.name] = (
                    sweep.parameter.instrument._channum
                )

            self.input_trigger = (
                {f"trigin_{input_trigger}": input_trigger} if input_trigger else None
            )
            self.output_trigger = (
                {f"trigout_{output_trigger}": output_trigger}
                if output_trigger
                else None
            )
            self.input_trigger_key = (
                f"trigin_{input_trigger}" if input_trigger else None
            )
            self.output_trigger_key = (
                f"trigout_{output_trigger}" if output_trigger else None
            )

            self.arrangement = self.core.arrange(
                contacts=self.contacts,
                output_triggers=self.output_trigger,
            )

            for trig in self.core.external_triggers:
                trig.width_s(self._trigger_width)

            if trigger_type == "ramp":
                raise NotImplementedError(
                    "trigger_type 'ramp' not implemented for 1D sweeps"
                )

            self._qdac_sweep = self.arrangement.virtual_sweep(
                contact=list(self.contacts.keys())[0],
                voltages=self.sweeps[0].values,
                start_sweep_trigger=self.input_trigger_key,
                step_time_s=self.delay,
                step_trigger=self.output_trigger_key,
            )

            return trigger_type, num_points, self.delay
        else:
            raise NotImplementedError("Only 1D and 2D sweeps are supported")

    def run_sweep(self):
        """Start the configured QDAC-II sweep and wait when this is the root."""
        self._sweep_active = True
        self._qdac_sweep.start()
        try:
            if self.toplevel:
                sleep((self.num + 1) * self.delay)
        finally:
            if self.toplevel:
                self.abort()

    def abort(self) -> None:
        """Stop the active QDAC list sweep and release its trigger routing."""
        if getattr(self, "_sweep_active", False):
            self._qdac_sweep.close()
            self._sweep_active = False

    def register_dependent(
        self,
        dependent: Parameter | Sequence[Parameter],
        num: int | Sequence[int],
        delay: float,
        input_trigger: int = 1,
    ) -> None:
        """
        Configure QDAC-II current measurements for the buffered sweep.

        Args:
            dependent (Parameter | Sequence[Parameter]): Current parameters to
                acquire.
            num (int | Sequence[int]): Point geometry supplied by the tree.
            delay (float): Measurement aperture in seconds.
            input_trigger (int): Accepted for the common node interface and
                ignored.

        Raises:
            NotImplementedError: If the QDAC-II is used only as an acquisition
                end node.

        """
        self._process_dependents(dependent)

        # TODO: Check that dependent is read_current_A, make it robust against parameter renames

        # A combined sweep/acquisition node uses its configured internal trigger.
        # Acquisition-only operation would require an external-trigger setup.
        if self.sweepnode:
            for dependent in self.dependents:
                dependent.instrument.clear_measurements()
                meas = dependent.instrument.measurement(aperture_s=delay)
                meas.start_on(
                    self.arrangement.get_trigger_by_name(self.output_trigger_key)
                )
        elif self.endnode:
            raise NotImplementedError("End node not implemented for QDAC2")

    def fetch(self) -> list[np.ndarray]:
        """
        Fetch buffered current readings from the QDAC-II.

        Returns:
            list[np.ndarray]: One flattened current array per dependent.

        """
        results = []
        for dependent in self.dependents:
            data = dependent.instrument.fetch_current_A()
            results.append(np.array(data).flatten())
        return results


class NodeKeysightVNA(BufferedNodeBase):
    """Keysight VNA node for buffered frequency sweeps."""

    def __init__(self, inst: Instrument, *args, **kwargs):
        super().__init__(inst=inst, *args, **kwargs)

    def register_sweep(
        self,
        sweep: Sweep | Sequence[Sweep],
        input_trigger: int | None = None,
        output_trigger: int | None = None,
        trigger_type: str | None = None,
        trigger_width: float | None = None,
    ) -> tuple[None, int, None]:
        """
        Configure a one-dimensional buffered frequency sweep.

        Args:
            sweep (Sweep | Sequence[Sweep]): One frequency sweep.
            input_trigger (int | None): Accepted for the common node interface
                and ignored.
            output_trigger (int | None): Accepted for the common node interface
                and ignored.
            trigger_type (str | None): Accepted for the common node interface
                and ignored.
            trigger_width (float | None): Accepted for the common node interface
                and ignored.

        Returns:
            tuple[None, int, None]: No trigger type, point count, and no fixed
                step time.

        Raises:
            NotImplementedError: If more than one sweep is supplied.

        """
        self._process_sweeps(sweep)

        if len(self.sweeps) != 1:
            raise NotImplementedError("Only one sweep supported")

        num_points = self.num
        sw = self.sweeps[0]
        self.core.frequency(sw.values)
        return None, num_points, None

    def register_dependent(
        self,
        dependent: Parameter | Sequence[Parameter],
        num: int | Sequence[int],
        delay: int | float,
        input_trigger: int | None = None,
    ) -> None:
        """
        Register S-parameter dependents to be measured by the VNA.

        Args:
            dependent (Parameter | Sequence[Parameter]): VNA parameters
                exposing an ``_sparam`` attribute.
            num (int | Sequence[int]): Expected point geometry.
            delay (int | float): Accepted for the common node interface and
                ignored.
            input_trigger (int | None): Accepted for the common node interface
                and ignored.

        """
        self._process_dependents(dependent)
        self.num = num

        self.core.configure_active_s_parameters(
            [dep._sparam for dep in self.dependents]
        )

    def run_sweep(self):
        """Run the configured VNA sweep."""
        self.core.traces[0].run_sweep()

    def fetch(self) -> list[np.ndarray]:
        """
        Fetch dependent data.

        Returns:
            list[np.ndarray]: One array per dependent.

        """
        return [dep() for dep in self.dependents]


class NodeBaselDAC(BufferedNodeBase):
    """Basel LNHR DAC node for one- or two-dimensional AWG sweeps."""

    def __init__(
        self,
        inst: Instrument,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize a node around a Basel LNHR DAC.

        Args:
            inst (Instrument): Basel LNHR DAC QCoDeS instrument.
            *args: Additional arguments passed to :class:`BufferedNodeBase`.
            **kwargs: Additional keyword arguments passed to
                :class:`BufferedNodeBase`.

        """
        super().__init__(inst=inst, *args, **kwargs)
        self.contacts = {}

    def register_sweep(
        self,
        sweep: Sweep | Sequence[Sweep],
        input_trigger: int | None = None,
        output_trigger: int | None = None,
        trigger_type: str | None = None,
        trigger_width: float | None = None,
    ) -> tuple[str | None, int, float]:
        """
        Configure a one- or two-dimensional DAC AWG sweep.

        Args:
            sweep (Sweep | Sequence[Sweep]): One or two channel sweeps.
            input_trigger (int | None): Accepted for the common node interface
                and ignored.
            output_trigger (int | None): Accepted for the common node interface
                and ignored.
            trigger_type (str | None): Trigger label returned to the tree.
            trigger_width (float | None): Accepted for the common node interface
                and ignored.

        Returns:
            tuple[str | None, int, float]: Trigger label, total point count, and
                innermost step time.

        Raises:
            ValueError: If the delay is shorter than 20 microseconds, or a 2D
                sweep does not place its channels on different AWGs.
            NotImplementedError: If more than two sweeps are supplied.

        """

        # The DAC requires at least 20 microseconds between ramp steps.
        minimum_delay = 20 * 1e-6  # 20 us

        self._process_sweeps(sweep)

        inner_sampling_rate = self.delay

        if self.delay < minimum_delay:
            raise ValueError(f"Delay is too small, use at least {minimum_delay} s")

        if len(self.sweeps) == 2:
            num_points = self.num

            inner_sweep = sweep[1]
            outer_sweep = sweep[0]

            inner_voltages = inner_sweep.values
            outer_voltages = outer_sweep.values

            self.contacts = {
                inner_sweep.parameter[0].name: inner_sweep.parameter[
                    0
                ].instrument._channum,
                outer_sweep.parameter[0].name: outer_sweep.parameter[
                    0
                ].instrument._channum,
            }

            inner_channel = self.contacts[inner_sweep.parameter[0].name]
            outer_channel = self.contacts[outer_sweep.parameter[0].name]

            if inner_channel < 13 and outer_channel > 12:
                inner_awg = self.core.awga
                outer_awg = self.core.awgc
            elif inner_channel > 12 and outer_channel < 13:
                inner_awg = self.core.awgc
                outer_awg = self.core.awga
            else:
                raise ValueError(
                    "Inner and outer channels must be on different AWGs (1-12 on AWG A, 13-24 on AWG C)"
                )

            # Configure both AWGs explicitly for the nested sweep.
            inner_awg.enable(False)
            outer_awg.enable(False)

            inner_awg.write_awg_config(
                {
                    "channel": inner_channel,
                    "cycles": len(outer_voltages),
                    "sampling_rate": inner_sampling_rate,
                    "waveform": inner_voltages,
                }
            )

            outer_awg.write_awg_config(
                {
                    "channel": outer_channel,
                    "cycles": 1,
                    "sampling_rate": inner_sampling_rate * len(inner_voltages),
                    "waveform": outer_voltages,
                }
            )

            inner_awg.trigger("disable")
            outer_awg.trigger("single step")

            self._BaselDAC_sweep_start = lambda: self.core.run_awg_sweep(
                [inner_awg, outer_awg]
            )

            return trigger_type, num_points, self.delay

        elif len(self.sweeps) == 1:
            num_points = self.num
            sweep = self.sweeps[0]

            self.contacts = {}
            if isinstance(sweep.parameter, Sequence):
                assert len(sweep.parameter) == 1, (
                    "Only one parameter supported for 1D sweeps"
                )
                for param in sweep.parameter:
                    self.contacts[param.name] = param.instrument._channum
            else:
                self.contacts[sweep.parameter.name] = (
                    sweep.parameter.instrument._channum
                )

            channel_number = self.contacts[list(self.contacts.keys())[0]]

            if channel_number < 13:
                awg = self.core.awga
            else:
                awg = self.core.awgc

            awg.enable(False)

            awg.write_awg_config(
                {
                    "channel": channel_number,
                    "cycles": 1,
                    "sampling_rate": inner_sampling_rate,
                    "waveform": sweep.values,
                }
            )

            awg.trigger("disable")

            self._BaselDAC_sweep_start = lambda: self.core.run_awg_sweep([awg])

            return trigger_type, num_points, self.delay
        else:
            raise NotImplementedError("Only 1D and 2D sweeps are supported")

    def run_sweep(self):
        """Start the configured Basel DAC AWG sweep."""
        self._BaselDAC_sweep_start()

    def register_dependent(
        self,
        dependent: Parameter | Sequence[Parameter],
        num: int | Sequence[int],
        delay: int | float,
        input_trigger: int | None = None,
    ) -> None:
        """
        Register parameters to read after the Basel DAC sweep.

        Args:
            dependent (Parameter | Sequence[Parameter]): Parameter or parameters
                read by :meth:`fetch`.
            num (int | Sequence[int]): Expected point geometry.
            delay (int | float): Step time supplied by the tree.
            input_trigger (int | None): Accepted for the common node interface
                and ignored.

        """
        self._process_dependents(dependent)
        self.num = num

    def fetch(self) -> list[np.ndarray]:
        """
        Fetch dependent data.

        Returns:
            list[np.ndarray]: One array per dependent.
        """
        return [dep() for dep in self.dependents]


class NodeDelay(BufferedNodeBase):
    """
    Virtual sweep node for buffered time sweeps.

    The node does not actively step anything. It defines the point count and
    spacing for a downstream buffered acquisition such as a Zurich Instruments
    DAQ module.

    """

    def __init__(self, inst: Instrument, *args, **kwargs):
        """
        Initialize a virtual delay node.

        Args:
            inst (Instrument): Placeholder instrument required by the common
                buffered-node interface.
            *args: Additional arguments passed to :class:`BufferedNodeBase`.
            **kwargs: Additional keyword arguments passed to
                :class:`BufferedNodeBase`.

        """
        super().__init__(inst=inst, *args, **kwargs)

    def register_sweep(
        self,
        sweep: Sweep | Sequence[Sweep],
        input_trigger: int | None = None,
        output_trigger: int | None = None,
        trigger_type: str | None = None,
        trigger_width: float | None = None,
        **kwargs,
    ) -> tuple[None, int, float]:
        """
        Register one time sweep without configuring physical hardware.

        Args:
            sweep (Sweep | Sequence[Sweep]): Sequence containing one sweep.
            input_trigger (int | None): Accepted for the common node interface
                and ignored.
            output_trigger (int | None): Accepted for the common node interface
                and ignored.
            trigger_type (str | None): Accepted for the common node interface
                and ignored.
            trigger_width (float | None): Accepted for the common node interface
                and ignored.
            **kwargs: Ignored compatibility options.

        Returns:
            tuple[None, int, float]: No trigger type, point count, and time
                spacing in seconds.

        Raises:
            ValueError: If more than one sweep is supplied.

        """
        self._process_sweeps(sweep)

        if len(self.sweeps) != 1:
            raise ValueError("NodeDelay only supports 1D time sweeps.")

        sw = self.sweeps[0]

        self.num = int(sw.num)
        self.delay = float(sw.delay)

        return None, self.num, self.delay

    def run_sweep(self):
        """
        Wait for the acquisition window when this delay node is the tree root.

        The downstream acquisition module runs asynchronously, so no physical
        parameter is stepped here.

        """
        if self.toplevel:
            sleep((self.num + 1) * self.delay)
