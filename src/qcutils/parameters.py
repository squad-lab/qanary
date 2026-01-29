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
    name : str, optional
        Parameter name. If None, a name is generated from the gate names.
    label : str, optional
        Display label. If None, a label describing the linear combination
        is generated automatically.
    **kwargs
        Additional arguments passed to ``Parameter``.
    """

    def __init__(
        self,
        gates: list[Parameter],
        factors: list[float],
        offsets: list[float],
        name: str | None = None,
        label: str | None = None,
        unit: str = "V",
        **kwargs,
    ):
        self.gates = gates
        self.factors = factors
        self.offsets = offsets

        root = _root_instrument(gates[0])
        if not all(_root_instrument(ch) is root for ch in gates):
            raise ValueError("All gates must belong to the same root instrument")

        if name is None:
            gate_names = "_".join(g.name for g in gates)
            name = f"virtual_gate_{gate_names}"

        if label is None:
            label = "Virtual Gate: "
            parts = [f"{f}*{g.label}" for f, g in zip(factors, gates)]
            label += " + ".join(parts)

        super().__init__(
            name=name,
            label=label,
            unit=unit,
            instrument=gates[0].instrument,
            set_cmd=None,
            get_cmd=None,
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

        snap["type"] = "virtual_gate_linear"
        snap["gates"] = {ch.name: ch.full_name for ch in self.gates}
        snap["transformation"] = {
            "factors": [float(f) for f in self.factors],
            "offsets": [float(o) for o in self.offsets],
        }

        return snap


class MultiChannelParameter(Parameter):
    """
    Parameter that groups multiple channels into a single control.

    Setting this parameter applies the same value to all channels.
    Getting it returns a tuple of the current channel values.

    Parameters
    ----------
    channels : Sequence[Parameter]
        Parameters controlled together.
    name : str, optional
        Parameter name. If None, an automatic name is generated.
    label : str, optional
        Display label. If None, a label is built from the channel labels.
    unit : str, optional
        Parameter unit. Defaults to "V".
    **kwargs
        Additional arguments passed to ``Parameter``.
    """

    def __init__(
        self,
        channels: Sequence[Parameter],
        name: str | None = None,
        label: str | None = None,
        unit: str = "V",
        **kwargs,
    ):
        root = _root_instrument(channels[0])
        if not all(_root_instrument(ch) is root for ch in channels):
            raise ValueError("All channels must belong to the same root instrument")

        if name is None:
            channel_names = "_".join(ch.name for ch in channels)
            name = f"multi_channel_parameter_{channel_names}"

        if label is None:
            label = "MultiChannelParameter: "
            parts = [f"{ch.label}" for ch in channels]
            label += " , ".join(parts)

        super().__init__(
            name=name,
            label=label,
            unit=unit,
            instrument=channels[0].instrument,
            vals=vals.Numbers(),
            **kwargs,
        )

        self.channels = list(channels)

    def set_raw(self, value: float):
        for ch in self.channels:
            ch(value)

    def get_raw(self):
        values = tuple(ch.get() for ch in self.channels)

        if all(v == values[0] for v in values):
            return float(values[0])
        else:
            raise ValueError(f"MultiChannelParameter values differ: {values}")

    def snapshot_base(self, update=False, params_to_skip_update=None):
        snap = super().snapshot_base(
            update=update,
            params_to_skip_update=params_to_skip_update,
        )
        snap["channels"] = [ch.full_name for ch in self.channels]
        return snap
