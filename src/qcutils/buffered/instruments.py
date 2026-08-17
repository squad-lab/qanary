from time import sleep, time
from typing import Sequence, Union

import numpy as np
from pyvisa.constants import StatusCode
from pyvisa.errors import VisaIOError
from qcodes.instrument import Instrument
from qcodes.parameters import Parameter

from qcutils.sweep import Sweep


class BufferedNodeBase:
    def __init__(self, inst: Instrument) -> None:
        """
        Base class for buffered sweep nodes. This class is a node of the buffered sweep tree. Can be used in three ways:
        1. As a node in a buffered sweep tree with a sweep and a dependent, where the device is both sweeping and measuring.
        2. As a node in a buffered sweep tree with a sweep and no dependent, where the device is only sweeping.
        3. As a node in a buffered sweep tree with no sweep and a dependent, where the device is only measuring.
        """
        self.buffered = True
        self.toplevel = False
        self._processed_sweeps = False
        self._processed_dependents = False
        self.core = inst

        self.endnode = True
        self.sweepnode = True

    def _process_sweeps(self, sweep: Sweep | Sequence[Sweep]):
        """
        Process sweeps

        Args:
            sweep (Sweep): The sweep object to be used in the buffered sweep tree.
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

    def _process_dependents(self, dependent: Parameter | Sequence[Parameter]):
        """
        Process dependents

        Args:
            dependent (Parameter | Sequence[Parameter]): The dependent object to be used in the buffered sweep tree.
        """
        if not isinstance(dependent, Sequence):
            self.dependents = [dependent]
        else:
            self.dependents = dependent

        # self.sweepnode = False

    def abort(self) -> None:
        """Stop active work owned by this buffered node, if any."""


class NodeMFLI(BufferedNodeBase):
    """
    MFLI node for a buffered sweep tree.

    - Uses the LabOne DAQ module in **hardware-trigger** mode (type=6).
    - Exact grid acquisition with:
        rows := number of trigger events (outer dimension)
        cols := number of points recorded per trigger (inner dimension)
        (choose grid mode)

    Typical flow:
        register_dependent(...)  # config + subscribe + execute()
        ... run your QDAC sweep to generate triggers ...
        fetch()                  # wait for completion and read data

    NOTE:
    Minimum trigger width is 1e-4s. Set the trigger width in the parent instrument accordingly.
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
        dependent: Union[Parameter, Sequence[Parameter]],
        num: int | Sequence[int],
        delay: float,
        input_trigger: int = 1,
        *,
        trigger_delay: float = 0.0,
        trigger_level: float = 0.5,
        tc_factor: float = 1.0,
        grid_mode: str = "linear",
        edge: str = "rising",
        endless: bool = False,
        count: int = 1,
    ) -> None:
        """
        Configure DAQ for a hardware-triggered, and start it.

        Args:
            dependent: LabOne node(s), e.g. "demods/0/sample.r" or list thereof.
            num: Number of points. For 2D, pass (rows, cols) where rows is the number
                 of trigger events (outer loop) and cols the points per trigger.
            delay: Step time (s) of the innermost sweep; DAQ duration ~ delay * cols.
            input_trigger: Which TrigIn (1 or 2) to use on the MFLI.
            trigger_delay: Delay (s) between trigger and first sample (default 0).
            tc_factor: Factor to adjust the time constant of the demodulator. The time constant is set to delay/tc_factor. (default 1)
            grid_mode: DAQ grid mode; 1=nearest, 2=linear, 4=exact grid (default).
            edge: Trigger edge; one of "rising", "falling", "both".
            endless: If True, run continuous acquisition (advanced use).
            count: Number of grids to acquire in single-shot mode (endless=False).
        """
        # Normalize dependents list

        if isinstance(dependent, Sequence):
            self.dependents = [dep.zi_node.lower() for dep in dependent]
        else:
            self.dependents = [dependent.zi_node.lower()]

        if grid_mode not in ["nearest", "linear", "exact"]:
            raise ValueError(
                f"Invalid grid_mode: {grid_mode}. Must be nearest, linear or exact."
            )

        # set grid mode
        self.daq_module.set("grid/mode", grid_mode)

        self.daq.setInt(f"/{self.serial}/demods/0/enable", 1)
        self.daq.setDouble(
            f"/{self.serial}/demods/0/timeconstant", float(delay) / tc_factor
        )  # control the time constant in relation to delay

        self.daq_module.finish()
        self.daq_module.unsubscribe("*")
        self.daq_module.set("clearhistory", 1)

        self.daq_module.set(
            "triggernode", f"/{self.serial}/demods/0/sample.TrigIn{int(input_trigger)}"
        )

        # trigger level
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

        # trigger delay - shift data points to end of each ramp step, trigger_delay governs deviations from that
        self.daq_module.set("delay", float(delay) + trigger_delay)

        self.daq_module.set("holdoff/time", max(0.0, float(delay) * (cols - 0.5)))

        self._subs = []
        for dep in self.dependents:
            path = f"/{self.serial}{dep}"
            self.daq_module.subscribe(path)
            self._subs.append(path)

        self.daq_module.execute()

    def fetch(self, *, timeout: float = 15.0) -> list[np.ndarray]:
        """
        Wait for DAQ completion and return a list of numpy arrays,
        one for each subscribed path in self._subs.
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
    UHFLI node for a buffered sweep tree - very similar to MFLI node

    - Uses the LabOne DAQ module in **hardware-trigger** mode (type=6).
    - Exact grid acquisition with:
        rows := number of trigger events (outer dimension)
        cols := number of points recorded per trigger (inner dimension)
        (choose grid mode)

    Typical flow:
        register_dependent(...)  # config + subscribe + execute()
        ... run your QDAC sweep to generate triggers ...
        fetch()                  # wait for completion and read data

    NOTE:
    Minimum trigger width is 1e-4s. Set the trigger width in the parent instrument accordingly.
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

        # Shared state per physical UHFLI
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
        sweep: Union[Sweep, Sequence[Sweep]],
        input_trigger: int = None,
        output_trigger: int = None,
        trigger_type: str = None,
        trigger_width: float = None,
        *,
        spacing: str = "lin",
        averaging: int = 1,
        averaging_tc: int = 5,
        sweep_order: int = 3,
        settling_inaccuracy: float = 100 * 1e-6,
        phase_unwrap: bool = False,
        **kwargs,
    ) -> None:
        """
        Configure and start a buffered sweep with the LabOne Sweeper Module. No triggering pissble/necessary
        You can sweep all eight demodulator frequencies and the eight amplitudes of both outputs.

        Parameters
        ----------
        sweep:
            qcutils Sweep object defining start, stop and num.
        spacing:
            "lin" = linear frequency axis (default), "log" = logarithmic.
        averaging:
            Number of averages for each sweep point.
        averaging_tc:
            Sets the effective number of time constants per sweeper parameter point that is considered in the measurement.
        sweep_order:
            Order of the sweep (filter roll off).
        settling_inaccuracy:
            Demodulator filter settling inaccuracy defining the wait time between a sweep parameter change and recording of the next sweep point.
        phase_unwrap:
            If True, the phase is unwrapped to avoid jumps of 2pi in the phase data.
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

        self.sweeper_module.set("scan", 0)  # sequential forward
        self.sweeper_module.set("xmapping", xmapping)

        self.sweeper_module.set("bandwidthcontrol", 2)  # Auto
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
        self.daq_module.finish()
        self.sweeper_module.execute()
        if self.toplevel:
            sleep((self.num + 1) * self.delay)

    def _register_daq_dependent(
        self,
        dependent: Union[Parameter, Sequence[Parameter]],
        num: int | Sequence[int],
        delay: float,
        input_trigger: int = 1,
        *,
        demod_channels: Union[int, Sequence[int]] = 0,
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
        Configure DAQ for a hardware-triggered, exact-grid acquisition and start it.

        Args:
            dependent: LabOne node(s), e.g. "demods/0/sample.r" or list thereof.
            demod_channels: Demodulator channels to use for the dependent(s).
            num: Number of points. For 2D, pass (rows, cols) where rows is the number
                 of trigger events (outer loop) and cols the points per trigger.
            delay: Step time (s) of the innermost sweep; DAQ duration ~ delay * cols.
            input_trigger: Which TrigIn (1 or 2) to use on the MFLI.
            trigger_delay: Delay (s) between trigger and first sample (default 0).
            tc_factor: Factor to adjust the time constant of the demodulator. The time constant is set to delay/tc_factor. (default 1)
            grid_mode: DAQ grid mode; 1=nearest, 2=linear, 4=exact grid (default).
            edge: Trigger edge; one of "rising", "falling", "both".
            endless: If True, run continuous acquisition (advanced use).
            count: Number of grids to acquire in single-shot mode (endless=False).
        """
        # Normalize dependents list

        if isinstance(dependent, Sequence):
            self.dependents = [dep.zi_node.lower() for dep in dependent]
        else:
            self.dependents = [dependent.zi_node.lower()]

        if grid_mode not in ["nearest", "linear", "exact"]:
            raise ValueError(
                f"Invalid grid_mode: {grid_mode}. Must be nearest, linear or exact."
            )

        # set grid mode
        self.daq_module.set("grid/mode", grid_mode)

        for demod in np.atleast_1d(demod_channels):
            self.daq.setInt(f"/{self.serial}/demods/{demod}/enable", 1)
            self.daq.setDouble(
                f"/{self.serial}/demods/{demod}/timeconstant", float(delay) / tc_factor
            )  # control the time constant in relation to delay

        self.daq_module.finish()
        self.daq_module.unsubscribe("*")
        self.daq_module.set("clearhistory", 1)

        for demod in np.atleast_1d(demod_channels):
            self.daq_module.set(
                "triggernode",
                f"/{self.serial}/demods/{demod}/sample.TrigIn{int(input_trigger)}",
            )

        # trigger level
        self.daq.setDouble(
            f"/{self.serial}/triggers/in/{int(input_trigger) - 1}/level",
            trigger_level,
        )

        # ensure input trigger impedances at 1 kOhm
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

        # trigger delay - shift data points to end of each ramp step, trigger_delay governs deviations from that
        self.daq_module.set("delay", float(delay) + trigger_delay)

        self.daq_module.set("holdoff/time", max(0.0, float(delay) * (cols - 0.5)))

        self._subs = []
        for dep in self.dependents:
            path = f"/{self.serial}{dep}"
            self.daq_module.subscribe(path)
            self._subs.append(path)

        self.daq_module.execute()

    def _register_sweeper_dependent(
        self,
        dependent: Union[Parameter, Sequence[Parameter]],
        num: int | Sequence[int],
        delay: float,
        input_trigger: int = None,
        **kwargs,
    ) -> None:
        if not self.endnode:
            raise ValueError("Sweeper dependents can only be registered on end nodes.")

        if isinstance(dependent, Sequence):
            dependents = list(dependent)
        else:
            dependents = [dependent]

        self.dependents = dependents

        # Mapping:
        # QCoDeS dependent -> sweeper sample field
        sweeper_fields = []

        # Subscribe each demod sample node only once
        sample_paths = set()

        for dep in dependents:
            zi_node = dep.zi_node.lower()

            # e.g.
            # /demods/0/sample.r
            # ->
            # /demods/0/sample
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
        dependent,
        num: int | list[int],
        delay,
        *,
        acquisition: str = "daq",
        **kwargs,
    ) -> None:
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

    def fetch(self, *, timeout: float = 15.0):
        if self._active_acquisition == "daq":
            return self._fetch_daq(timeout=timeout)

        if self._active_acquisition == "sweeper":
            return self._fetch_sweeper(timeout=2 * timeout)

        raise RuntimeError("No acquisition has been registered.")

    def _fetch_daq(self, *, timeout: float = 15.0) -> list[np.ndarray]:
        """
        Wait for DAQ completion and return a list of numpy arrays,
        one for each subscribed path in self._subs.
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

            # Example:
            # /dev2793/demods/0/sample
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
    Keysight 344xxA DMM node for a buffered sweep tree.

    - Uses the DMM's internal reading memory.
    - Acquisition is triggered by an external trigger source (EXT).
    - Supports 1D and 2D buffered acquisitions:
        rows := number of points acquired per trigger (inner dimension)
        cols := number of trigger events (outer dimension)

    Typical flow:
        register_dependent(...)  # configure triggering, timing, and integration
        ... run external sweep device to generate triggers ...
        fetch()                  # read out buffered data

    NOTE:
    Minimum trigger width is 1e-3s. Set the trigger width in the parent instrument accordingly.
    """

    def __init__(self, inst: Instrument, *args, **kwargs) -> None:
        super().__init__(inst=inst, *args, **kwargs)
        self.core.trigger.source("IMM")
        self.core.reset()

    def register_dependent(
        self,
        dependent: Union[Parameter, Sequence[Parameter], None],
        num: int | Sequence[int],
        delay: int | float,
        input_trigger: int = 1,
        trigger_type: str = "step",
    ) -> None:
        """
        Configure the Keysight DMM for a buffered, externally triggered acquisition.

        Args:
            dependent (Parameter | Sequence[Parameter] | None):
                Dependent parameter(s) associated with this node.
            num (int | Sequence[int]):
                Number of points. For 2D acquisitions, pass (rows, cols) where:
                    cols := points acquired per trigger
                    rows := number of trigger events
            delay (int | float):
                Step time in seconds. Used for integration time and sample timing.
            input_trigger (int):
                External trigger selector (kept for interface compatibility).
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
            rows, cols = int(num[0]), int(num[1])
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
    def __init__(
        self,
        inst: Instrument,
        *args,
        **kwargs,
    ) -> None:
        """
        QDAC2 as a node in the buffered sweep tree

        Args:
            sweep (Sweep or Sequence[Sweep]): The sweep object to be used in the buffered sweep tree. Can be a 1D or 2D sweep.
            dependent (Union[Parameter, Sequence[Parameter]]): Dependent parameter to be measured. Only read_current_A is supported.
        """
        super().__init__(inst=inst, *args, **kwargs)
        self.contacts = {}

    def register_sweep(
        self,
        sweep: Union[Sweep, Sequence[Sweep]],
        input_trigger: int = None,
        output_trigger: int = None,
        trigger_type: str = "ramp",
        trigger_width: float = 1e-4,
    ):
        """
        Register the 1D/2D buffered sweep with triggers for the QDAC2

        Args:
            sweep (Union[Sweep, Sequence[Sweep]]): qcutils Sweep or list of Sweeps
            input_trigger (int): input trigger for the QDAC2 (Optional, defaults to None)
            output_trigger (int): output trigger for the QDAC2 (Optional, defaults to None)
            trigger_type (str): Type of trigger for the QDAC2, only to be used for 2D sweeps. One of either "ramp" (trigger at the start of each ramp) or "step" (at each step). (Optional, defaults to "ramp" for 2D sweeps, "step" for 1D sweeps)

        Returns:
            num_points (int): Number of points in the sweep
            step_time (int | float): Duration of the innermost sweep in seconds. num_points * step_time = total time of the whole sweep sequence
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
        dependent: Union[Parameter, Sequence[Parameter]],
        num: int | Sequence[int],
        delay: float,
        input_trigger: int = 1,
    ) -> None:
        """
        Register the measurement with the QDAC2

        Args:
            step_time (int or float): Duration of the current measurement in seconds. Depends on the preceeding sweep's step size.
        """
        self._process_dependents(dependent)

        # TODO: Check that dependent is read_current_A, make it robust against parameter renames

        # if the device has sweeps set, then trigger by internal trigger
        # if the device is acting as an end node, then trigger by external trigger
        if self.sweepnode:
            for dependent in self.dependents:
                dependent.instrument.clear_measurements()
                meas = dependent.instrument.measurement(aperture_s=delay)
                meas.start_on(
                    self.arrangement.get_trigger_by_name(self.output_trigger_key)
                )
        elif self.endnode:
            raise NotImplementedError("End node not implemented for QDAC2")

    def fetch(self) -> list:
        """
        Fetch the measurement from the QDAC2

        Returns:
            list: List of the measured values
        """
        results = []
        for dependent in self.dependents:
            data = dependent.instrument.fetch_current_A()
            results.append(np.array(data).flatten())
        return results


class NodeKeysightVNA(BufferedNodeBase):
    """
    Keysight VNA node for a buffered sweep tree.
    """

    def __init__(self, inst: Instrument, *args, **kwargs):
        super().__init__(inst=inst, *args, **kwargs)

    def register_sweep(
        self,
        sweep: Sweep,
        input_trigger: int = None,
        output_trigger: int = None,
        trigger_type: str = None,
        trigger_width: float = None,
    ):
        """
        Register a buffered frequency sweep on the VNA.

        Args:
            sweep (Sweep): qcutils Sweep (only 1D supported).
            input_trigger (int, optional): Not supported.
            output_trigger (int, optional): Not supported.
            trigger_type (str, optional): Not supported.
            trigger_width (float, optional): Not supported.

        Returns:
            tuple: (trigger_type, num_points, step_time). Step time is None.
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
        dependent: Parameter | list[Parameter] | None,
        num: int | list[int],
        delay: int | float,
        input_trigger: int = None,
    ) -> None:
        """
        Register S-parameter dependents to be measured by the VNA.

        Args:
            dependent (Parameter | list[Parameter] | None): Dependent parameter(s)
                created by the VNA driver (must have `._sparam` attribute).
            num (int | list[int]): Number of points expected (stored for fetch).
            delay (int | float): Step time (unused by the VNA node).
            input_trigger (int, optional): Not supported.
        """
        self._process_dependents(dependent)
        self.num = num

        self.core.configure_active_s_parameters(
            [dep._sparam for dep in self.dependents]
        )

    def run_sweep(self):
        """Run the VNA sweep."""
        self.core.traces[0].run_sweep()

    def fetch(self):
        """
        Fetch dependent data.

        Returns:
            list[np.ndarray]: One array per dependent.
        """
        return [dep() for dep in self.dependents]


class NodeBaselDAC(BufferedNodeBase):
    def __init__(
        self,
        inst: Instrument,
        *args,
        **kwargs,
    ) -> None:
        """
        BaselDAC as a node in the buffered sweep tree, it can oly be on top of the tree

        trigger setup: for 1D sweep just diable, for 2D sweep use awg a and c - put BNC sync out A into Trig in C and vice versa - set trigger type of inner awg to disable and trigger type of outer awg to single step

        Args:
            sweep (Sweep or Sequence[Sweep]): The sweep object to be used in the buffered sweep tree. Can be a 1D or 2D sweep.
            dependent (Union[Parameter, Sequence[Parameter]]): Dependent parameter to be measured. E.g. MFLI or UHFLI
        """
        super().__init__(inst=inst, *args, **kwargs)
        self.contacts = {}

    def register_sweep(
        self,
        sweep: Union[Sweep, Sequence[Sweep]],
        input_trigger: int = None,
        output_trigger: int = None,
        trigger_type: str = None,
        trigger_width: float = None,
    ):
        """
        Register the 1D/2D buffered sweep with triggers for the LNHR DAC 2 - make sure to have correct bandwidth (high or low) on used channels!

        Args:
            sweep (Union[Sweep, Sequence[Sweep]]): qcutils Sweep or list of Sweeps
            trigger_type (str): Type of trigger for Basel dac
        Returns:
            num_points (int): Number of points in the sweep
            step_time (int | float): Duration of the innermost sweep in seconds. num_points * step_time = total time of the whole sweep sequence
        """

        # minimum delay you can have between steps in a ramp
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

            # manually write the awg arrangement for the BaselDAC
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
            )  # if it does not work only enable first inner awg!

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
        self._BaselDAC_sweep_start()

    def register_dependent(
        self,
        dependent: Parameter | list[Parameter] | None,
        num: int | list[int],
        delay: int | float,
        input_trigger: int = None,
    ) -> None:
        """
        Args:
            dependent (Parameter | list[Parameter] | None): Dependent parameter(s)
                created by the VNA driver (must have `._sparam` attribute).
            num (int | list[int]): Number of points expected (stored for fetch).
        """
        self._process_dependents(dependent)
        self.num = num

    def fetch(self):
        """
        Fetch dependent data.

        Returns:
            list[np.ndarray]: One array per dependent.
        """
        return [dep() for dep in self.dependents]
