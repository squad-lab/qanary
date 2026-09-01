from qcodes import Parameter
from qcodes import validators as vals

from typing import Sequence
import numpy as np


### helper ###


def _root_instrument(param: Parameter):
    instr = param.instrument
    while hasattr(instr, "parent") and instr.parent is not None:
        instr = instr.parent
    return instr


##############


class ParameterMixin:
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
    Parameter representing a linear virtual gate composed of multiple gates.

    Setting this parameter applies a linear transformation (like a rotation) to each underlying
    gate using predefined factors and offsets. Getting the parameter returns
    the current values of all underlying gates.

    Parameters
    ----------
    gates : list[Parameter]
        Underlying gate parameters controlled by the virtual gate.
    factors : list[float]
        Linear scaling factors applied to the virtual gate value.
    offsets : list[float]
        Offsets added after scaling for each gate.
    rot_angle_deg : float, optional
        If provided, overrides factors and offsets to represent a rotation by this angle in degrees counter clockwise. first gate is x direction, second gate is y direction. Offsets are set to zero.
    points : list[tuple[float, float]], optional
        If provided, overrides factors and offsets to represent a virtual gate defined by two points in the 2D space of the first two gates. The first point defines the offset, and the direction from the first to the second point defines the factors. voltage steps along the virtual gate correspond to the distance between the two points.
    name : str, optional
        Parameter name. If None, a name is generated from the gate names.
    label : str, optional
        Display label. If None, a label describing the linear combination
        is generated automatically.
    param_type : str
        Type of the parameter, default is "virtual_gate_linear".
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
        for ch, f, o in zip(self.gates, self.factors, self.offsets):
            ch(f * value + o)

    def get_raw(self) -> float:
        vals = tuple(ch.get() for ch in self.gates)
        return vals

    def snapshot_base(self, update=False, params_to_skip_update=None):
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
    def __init__(
        self,
        param: Sequence[Parameter],
        name: str = None,
        label: str = None,
        param_type: str = "gates",
        **kwargs,
    ):
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
        for ch in self.channels:
            ch(value)

    def get_raw(self):
        values = tuple(ch.get() for ch in self.channels)

        if all(v == values[0] for v in values):
            return float(values[0])

        return None

    def snapshot_base(
        self,
        update=False,
        params_to_skip_update=None,
    ):
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
