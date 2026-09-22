# %%

from math import isclose, prod
from time import sleep
from typing import Sequence

import numpy as np
from qcdrivers.buffered._typing import SweepLike
from qcdrivers.buffered.base import BufferedNodeBase
from qcdrivers.squad.helpers import ShellInstrument
from qcodes.instrument import Instrument

from qanary.live_tuning.live_tuning import LiveTuning
from qanary.measure import Station
from qanary.sweep import Sweep

# %%


class NodeDummySweeper(BufferedNodeBase):
    """
    Virtual sweep node for buffered 1D and 2D sweeps.

    The node does not actively step physical hardware. It defines the sweep
    shape and point spacing for a downstream buffered acquisition node.

    Supported:
        1D: (n,)
        2D: (n_outer, n_inner)

    More than two sweep dimensions are rejected.
    """

    def __init__(self, inst: Instrument, *args, **kwargs):
        super().__init__(inst=inst, *args, **kwargs)

        self.shape: tuple[int, ...] = ()
        self.num: int | tuple[int, int] = 0
        self.total_num: int = 0
        self.delay: float = 0.0

    def register_sweep(
        self,
        sweep: SweepLike | Sequence[SweepLike],
        input_trigger: int | None = None,
        output_trigger: int | None = None,
        trigger_type: str | None = None,
        trigger_width: float | None = None,
        **kwargs,
    ) -> tuple[
        None,
        int | tuple[int, int],
        float,
    ]:
        """
        Register a virtual buffered 1D or 2D sweep.

        Args:
            sweep:
                One sweep or a sequence containing one or two sweeps.

                For two sweeps the ordering is preserved:

                    sweep[0] -> outer dimension
                    sweep[1] -> inner dimension

            input_trigger:
                Accepted for compatibility and ignored.

            output_trigger:
                Accepted for compatibility and ignored.

            trigger_type:
                Accepted for compatibility and ignored.

            trigger_width:
                Accepted for compatibility and ignored.

            **kwargs:
                Ignored compatibility options.

        Returns:
            tuple:
                ``(None, num, delay)``

                For 1D:
                    ``num`` is an int.

                For 2D:
                    ``num`` is ``(n_outer, n_inner)``.

                ``delay`` is the common sampling interval.

        Raises:
            ValueError:
                If zero sweeps, more than two sweeps, or different delays
                for the two dimensions are supplied.
        """
        self._process_sweeps(sweep)

        ndim = len(self.sweeps)

        if ndim == 0:
            raise ValueError("NodeDelay requires at least one sweep.")

        if ndim > 2:
            raise ValueError("NodeDelay supports at most 2D buffered sweeps.")

        # Preserve the sweep order:
        #
        #   sweeps[0] -> outer dimension
        #   sweeps[1] -> inner dimension
        self.shape = tuple(int(sw.num) for sw in self.sweeps)

        self.total_num = prod(self.shape)

        delays = tuple(float(sw.delay) for sw in self.sweeps)

        if ndim == 2 and not isclose(
            delays[0],
            delays[1],
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "NodeDelay requires the same delay for both "
                "dimensions of a 2D buffered sweep. "
                f"Received {delays[0]} s and {delays[1]} s."
            )

        self.delay = delays[0]

        # Keep the old 1D interface backwards compatible.
        if ndim == 1:
            self.num = self.shape[0]
        else:
            self.num = (
                self.shape[0],
                self.shape[1],
            )

        return None, self.num, self.delay

    def run_sweep(self) -> None:
        """
        Wait for the complete acquisition window when this node is the
        root of the buffered tree.

        For 2D the sweep is treated as one flattened acquisition buffer:

            total_num = n_outer * n_inner
        """
        if not self.toplevel:
            return

        if self.total_num <= 0:
            raise RuntimeError("No sweep has been registered on NodeDelay.")

        sleep((self.total_num + 1) * self.delay)


class NodeDummyAcquisition(BufferedNodeBase):
    def __init__(
        self,
        inst,
        noise: float = 0.0,
        seed: int | None = 42,
        *args,
        **kwargs,
    ):
        super().__init__(
            inst=inst,
            *args,
            **kwargs,
        )

        self.noise = noise
        self.rng = np.random.default_rng(seed)

        # Important: define these already here.
        self.shape: tuple[int, ...] = ()
        self.total_num: int = 0
        self.delay: float = 0.0

        self.frame = 0

    def register_dependent(
        self,
        dependent,
        num,
        delay,
        **kwargs,
    ):
        """
        Register buffered dependents and remember the sweep shape.

        num:
            1D: int, e.g. 101
            2D: tuple/list, e.g. (51, 101)
        """

        self._process_dependents(dependent)

        if isinstance(num, Sequence) and not isinstance(
            num,
            (str, bytes),
        ):
            self.shape = tuple(int(n) for n in num)
        else:
            self.shape = (int(num),)

        if len(self.shape) not in (1, 2):
            raise ValueError(
                "NodeDummyAcquisition only supports "
                f"1D or 2D sweeps, got shape={self.shape}."
            )

        self.total_num = int(np.prod(self.shape))

        self.delay = float(delay)

    def fetch(self) -> list[np.ndarray]:

        gate_x = float(self.core.gate_x())
        gate_y = float(self.core.gate_y())

        if len(self.shape) != 2:
            raise RuntimeError("This dummy acquisition currently expects a 2D sweep.")

        ny, nx = self.shape

        # IMPORTANT:
        # These ranges must match the actual sweep coordinates.
        x = np.linspace(
            -1.0,
            1.0,
            nx,
        )

        y = np.linspace(
            -1.0,
            1.0,
            ny,
        )

        X, Y = np.meshgrid(
            x,
            y,
            indexing="xy",
        )

        # The controls now directly use the same coordinate system
        # as the plotted sweep axes.
        x0 = gate_x
        y0 = gate_y

        sigma = 0.15

        gaussian = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2))

        if self.noise > 0:
            gaussian += self.noise * self.rng.standard_normal(gaussian.shape)

        r_data = gaussian
        p_data = 30.0 * gaussian

        result = []

        for dependent in self.dependents:
            if dependent.name == "lockin_r":
                data = r_data

            elif dependent.name == "lockin_p":
                data = p_data

            else:
                raise ValueError(f"Unknown dummy dependent {dependent.name!r}.")

            result.append(np.asarray(data).ravel())

        self.frame += 1

        return result


# %%

dummy_dac = ShellInstrument("dummy_dac", {})
dummy_acquisition = ShellInstrument("dummy_lockin", {})

# %%

dummy_acquisition.add_parameter(
    "gate_x",
    label="Dummy Gate X",
    unit="V",
    initial_value=0,
    get_cmd=None,
    set_cmd=None,
)

dummy_acquisition.add_parameter(
    "gate_y",
    label="Dummy Gate Y",
    unit="V",
    initial_value=0,
    get_cmd=None,
    set_cmd=None,
)

dummy_acquisition.add_parameter(
    "R",
    label="Dummy R",
    unit="V",
    get_cmd=lambda: 0.0,
)

dummy_acquisition.add_parameter(
    "P",
    label="Dummy Phase",
    unit="deg",
    get_cmd=lambda: 0.0,
)

# %%

dummy_dac.add_parameter(
    "V1",
    label="Dummy DAC 1",
    unit="V",
    set_cmd=lambda x: None,
)

dummy_dac.add_parameter(
    "V2",
    label="Dummy DAC 2",
    unit="V",
    set_cmd=lambda x: None,
)

dummy_dac.add_parameter(
    "V3",
    label="Dummy DAC 3",
    unit="V",
    set_cmd=lambda x: None,
)

# %%

st = Station("Test")
st.instruments = [dummy_dac, dummy_acquisition]

# %%

run_dict = {
    "wafer_id": "wafer1",
    "device_type": "QD",
    "sample_name": "device1",
    "station": st,
    "experiment_name": "live_tuning",
    "data_location": r"C:\Users\s.schreibing\Documents\qimchi_data",
    "fridge_name": "Radler",
}

# %%

v1 = st.add_parameter("V1", "DAC 1", dummy_dac.V1)
v2 = st.add_parameter("V2", "DAC 2", dummy_dac.V2)
v3 = st.add_parameter("V3", "DAC 3", dummy_dac.V3)

lockin_r = st.add_parameter("lockin_r", "Lock-in R", dummy_acquisition.R)
lockin_p = st.add_parameter("lockin_p", "Lock-in Phase", dummy_acquisition.P)

dummy_gate_x = st.add_parameter(
    "dummy_gate_x",
    "Dummy Gate X",
    dummy_acquisition.gate_x,
)

dummy_gate_y = st.add_parameter(
    "dummy_gate_y",
    "Dummy Gate Y",
    dummy_acquisition.gate_y,
)


# %%

v1_sweep = Sweep(v1, -1, 1, num=51, delay=0.001, start_delay=0.001)
v2_sweep = Sweep(v2, -1, 1, num=51, delay=0.001, start_delay=0.001)

buffered_sweep = {
    "instrument": NodeDummySweeper(inst=dummy_dac),
    "sweeps": [v1_sweep, v2_sweep],
    "nodes": [
        {
            "instrument": NodeDummyAcquisition(
                inst=dummy_acquisition,
                noise=0.02,
            ),
            "dependent": [
                lockin_p,
                lockin_r,
            ],
        }
    ],
}

# %%

tuning = LiveTuning(
    controls=[
        dummy_gate_x,
        dummy_gate_y,
    ],
    control_ranges=[
        (-1.0, 1.0),
        (-1.0, 1.0),
    ],
    refresh_interval=0.1,
)

# %%

tuning.start([buffered_sweep], **run_dict, no_hashing=True)

# %%

tuning.stop()

# %%

Instrument.close_all()

# %%
