from typing import Sequence

import numpy as np
from qcodes import Parameter
from qcodes import validators as vals

# Public API
__all__ = [
    "ParameterMixin",
    "VirtualGate",
    "MultiChannelParameter",
]


def _root_instrument(param: Parameter):
    instr = param.instrument
    while hasattr(instr, "parent") and instr.parent is not None:
        instr = instr.parent
    return instr


class ParameterMixin:
    """
    Alias an existing QCoDeS parameter for station metadata.

    The returned object keeps the original parameter class and state while
    optionally replacing its short name and label.

    Args:
        param (Parameter): Existing QCoDeS parameter.
        name (str): Optional short name override.
        label (str): Optional display-label override.
        param_type (str): Category stored in the parameter snapshot. Defaults
            to ``"gate"``.

    """

    def __init__(
        self,
        param: Parameter,
        name: str = None,
        label: str = None,
        param_type: str = "gate",
    ):
        self.__dict__.update(param.__dict__)
        self.__class__ = param.__class__

        self.param_type = param_type
        if name:
            self._short_name = name
        if label:
            self._label = label


class VirtualGate(Parameter):
    """
    Linear virtual coordinate composed from parameters on one instrument.

    """

    def __init__(
        self,
        gates: list[Parameter],
        factors: list[float] = None,
        offsets: list[float] = None,
        rot_angle_deg: float | None = None,
        points: list[tuple[float, float]] = None,
        name: str | None = None,
        label: str | None = None,
        param_type: str = "virtual_gate_linear",
        **kwargs,
    ):
        """
        Create a virtual gate from coefficients, a rotation, or two points.

        Setting virtual value ``v`` writes ``factor * v + offset`` to each
        underlying gate. Reading returns a tuple containing every physical gate
        value.

        Args:
            gates (list[Parameter]): Underlying gate parameters. They must share
                one root instrument.
            factors (list[float]): Scale factor for each gate.
            offsets (list[float]): Offset for each gate.
            rot_angle_deg (float | None): For exactly two gates, replace
                *factors* with the unit vector at this counter-clockwise angle
                and set both offsets to zero.
            points (list[tuple[float, float]]): For exactly two gates, use the
                first point as the offset and the normalized direction to the
                second point as the factors. This takes precedence when supplied
                together with a rotation.
            name (str | None): QCoDeS parameter name. Generated from gate names
                when omitted.
            label (str | None): Display label. Generated from gate names when
                omitted.
            param_type (str): Snapshot category. Defaults to
                ``"virtual_gate_linear"``.
            **kwargs: Additional keyword arguments passed to QCoDeS
                :class:`~qcodes.parameters.Parameter`.

        Raises:
            ValueError: If coefficient lengths differ from the gate count, a
                two-dimensional definition does not receive exactly two gates,
                the two points coincide, or gates have different roots.

        """
        self.gates = gates
        self.param_type = param_type

        self.factors = factors
        self.offsets = offsets

        self.points = points
        self.rot_angle_deg = rot_angle_deg

        if self.factors is not None and self.offsets is not None:
            if len(gates) != len(self.factors) or len(gates) != len(self.offsets):
                raise ValueError(
                    "The number of gates, factors, and offsets must be the same"
                )

        if rot_angle_deg is not None:
            if len(self.gates) != 2:
                raise ValueError("Rotation is only supported for exactly two gates")
            theta = np.deg2rad(rot_angle_deg)
            self.factors = [np.cos(theta), np.sin(theta)]
            self.offsets = [0, 0]

        if self.points is not None:
            if len(self.gates) != 2:
                raise ValueError(
                    "Point-based virtual gates are only supported for exactly two gates"
                )

            P1 = np.asarray(self.points[0], dtype=float)
            P2 = np.asarray(self.points[1], dtype=float)

            direction = P2 - P1
            distance = np.linalg.norm(direction)

            if distance == 0:
                raise ValueError("The two points must be different")

            direction /= distance

            self.factors = direction.tolist()
            self.offsets = P1.tolist()

        root = _root_instrument(gates[0])
        if not all(_root_instrument(ch) is root for ch in gates):
            raise ValueError("All gates must belong to the same root instrument")

        if name is None:
            gate_names = "_".join(g.name for g in gates)
            name = f"virtual_gate_{gate_names}"

        if label is None:
            gate_names = " ".join(g.name for g in gates)
            label = f"Virtual Gate {gate_names}"

        unit = gates[0].unit

        super().__init__(
            name=name,
            label=label,
            unit=unit,
            instrument=root,
            vals=vals.Numbers(),
            **kwargs,
        )

    def set_raw(self, value: float):
        """Apply the configured linear transformation to every gate."""
        for ch, f, o in zip(self.gates, self.factors, self.offsets):
            ch(f * value + o)

    def get_raw(self) -> float:
        """Return the current values of all underlying gates as a tuple."""
        vals = tuple(ch.get() for ch in self.gates)
        return vals

    def snapshot_base(self, update=False, params_to_skip_update=None):
        """Extend the QCoDeS snapshot with the virtual-gate transformation."""
        snap = super().snapshot_base(
            update=update,
            params_to_skip_update=params_to_skip_update,
        )

        snap["param_type"] = self.param_type
        snap["gates"] = {ch.name: ch.full_name for ch in self.gates}
        snap["transformation"] = {
            "factors": [float(f) for f in self.factors],
            "offsets": [float(o) for o in self.offsets],
        }

        parts = [
            f"{g.label} = {o:.3f} + {f:.3f} * V_virtual"
            for g, f, o in zip(self.gates, self.factors, self.offsets)
        ]
        snap["description"] = "Virtual Gate: [" + ", ".join(parts) + "]"

        snap["rot_angle_deg"] = self.rot_angle_deg
        snap["points"] = self.points

        return snap


class MultiChannelParameter(Parameter):
    """
    Expose several channels on one instrument as a single parameter.

    """

    def __init__(
        self,
        param: Sequence[Parameter],
        name: str = None,
        label: str = None,
        param_type: str = "gates",
        **kwargs,
    ):
        """
        Create a parameter that writes the same value to every channel.

        Args:
            param (Sequence[Parameter]): One or more channels sharing a root
                instrument.
            name (str): Optional parameter name generated from channel names
                when omitted.
            label (str): Optional label generated from channel labels when
                omitted.
            param_type (str): Snapshot category. Defaults to ``"gates"``.
            **kwargs: Additional keyword arguments passed to QCoDeS
                :class:`~qcodes.parameters.Parameter`.

        Raises:
            ValueError: If no channels are supplied or their root instruments
                differ.

        """
        channels = list(param)

        if not channels:
            raise ValueError("At least one channel must be provided")

        root = _root_instrument(channels[0])
        if not all(_root_instrument(ch) is root for ch in channels):
            raise ValueError("All channels must belong to the same root instrument")

        if name is None:
            channel_names = "_".join(ch.name for ch in channels)
            name = f"multi_channel_parameter_{channel_names}"

        if label is None:
            label = "MultiChannelParameter: " + ", ".join(ch.label for ch in channels)

        unit = channels[0].unit

        super().__init__(
            name=name,
            label=label,
            unit=unit,
            instrument=root,
            vals=vals.Numbers(),
            **kwargs,
        )

        self.channels = channels
        self.param_type = param_type

    def set_raw(self, value: float):
        """
        Write *value* to every channel.

        """
        for ch in self.channels:
            ch(value)

    def get_raw(self):
        """
        Return the common channel value, or ``None`` if values differ.

        """
        values = tuple(ch.get() for ch in self.channels)

        if all(v == values[0] for v in values):
            return float(values[0])

        return None

    def snapshot_base(
        self,
        update=False,
        params_to_skip_update=None,
    ):
        """
        Extend the QCoDeS snapshot with channel identities and type.

        """
        snap = super().snapshot_base(
            update=update,
            params_to_skip_update=params_to_skip_update,
        )

        snap["param_type"] = self.param_type

        snap["channels"] = [
            {
                "name": ch.full_name,
                "label": ch.label,
                "unit": ch.unit,
            }
            for ch in self.channels
        ]

        return snap
