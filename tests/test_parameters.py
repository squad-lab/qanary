"""
Tests for the composite parameters in ``qcutils.parameters``.

``VirtualGate`` and ``MultiChannelParameter`` both present several physical
gates as one sweepable knob, so the same failure mode applies to both: a
mis-built composite silently drives the wrong voltages onto a device. The
constructors therefore validate eagerly, and these tests pin every rejection
alongside the transformations they compute.

``VirtualGate`` registers itself on its root instrument, so each test passes an
explicit ``name`` -- QCoDeS rejects a duplicate parameter name on an
instrument, and the generated names collide as soon as two virtual gates share
their underlying gates.

"""

import numpy as np
import pytest

from qcutils.parameters import (
    MultiChannelParameter,
    ParameterMixin,
    VirtualGate,
    _root_instrument,
)


class TestRootInstrument:
    """``_root_instrument`` decides whether two gates may be combined."""

    def test_returns_the_instrument_of_a_flat_parameter(self, gates):
        assert _root_instrument(gates.x) is gates

    def test_walks_up_from_a_channel_to_its_parent(self, instrument):
        """A channel parameter belongs to the channel, not the instrument."""
        inst = instrument(channels=True)
        channel_param = inst.channels[0].temperature

        assert channel_param.instrument is not inst
        assert _root_instrument(channel_param) is inst


class TestParameterMixin:
    """``ParameterMixin`` relabels a parameter without wrapping it."""

    def test_overrides_name_and_label_while_keeping_the_class(self, gates):
        aliased = ParameterMixin(gates.x, name="plunger", label="Plunger 1")

        assert aliased.name == "plunger"
        assert aliased.label == "Plunger 1"
        assert isinstance(aliased, type(gates.x))

    def test_keeps_reading_the_original_parameter(self, gates):
        """
        The mixin copies the parameter's ``__dict__``, cache included, so it
        must report values set through the original gate.

        """
        aliased = ParameterMixin(gates.x, name="plunger")
        gates.x(0.75)

        assert aliased() == pytest.approx(0.75)

    def test_defaults_to_a_gate_and_keeps_the_original_names(self, gates):
        aliased = ParameterMixin(gates.x)

        assert aliased.param_type == "gate"
        assert aliased.name == gates.x.name
        assert aliased.label == gates.x.label


class TestVirtualGateConstruction:
    """Every way of specifying the linear transformation, and its rejections."""

    def test_explicit_factors_and_offsets_are_kept(self, gates):
        virtual = VirtualGate(
            [gates.x, gates.y],
            factors=[1.0, 2.0],
            offsets=[0.1, 0.2],
            name="vg_explicit",
        )

        assert virtual.factors == [1.0, 2.0]
        assert virtual.offsets == [0.1, 0.2]

    @pytest.mark.parametrize(
        ("factors", "offsets"),
        [([1.0], [0.0, 0.0]), ([1.0, 1.0], [0.0]), ([1.0], [0.0])],
    )
    def test_rejects_factors_or_offsets_of_the_wrong_length(
        self, gates, factors, offsets
    ):
        with pytest.raises(ValueError, match="must be the same"):
            VirtualGate(
                [gates.x, gates.y],
                factors=factors,
                offsets=offsets,
                name="vg_bad_length",
            )

    @pytest.mark.parametrize(
        ("angle", "expected"),
        [(0.0, (1.0, 0.0)), (90.0, (0.0, 1.0)), (45.0, (0.5**0.5, 0.5**0.5))],
    )
    def test_rotation_angle_builds_a_unit_vector(self, gates, angle, expected):
        """The first gate is x, the second y, and offsets are zeroed."""
        virtual = VirtualGate(
            [gates.x, gates.y], rot_angle_deg=angle, name=f"vg_rot_{int(angle)}"
        )

        assert virtual.factors == pytest.approx(expected, abs=1e-12)
        assert virtual.offsets == [0, 0]

    def test_rotation_requires_exactly_two_gates(self, instrument):
        inst = instrument("a", "b", "c")

        with pytest.raises(ValueError, match="exactly two gates"):
            VirtualGate(
                [inst.a, inst.b, inst.c], rot_angle_deg=30.0, name="vg_rot_three"
            )

    def test_two_points_define_direction_and_origin(self, gates):
        """
        A 3-4-5 triangle: the factors are the normalised direction and the
        offsets are the first point, so one unit of the virtual gate moves the
        pair one unit along the line.

        """
        virtual = VirtualGate(
            [gates.x, gates.y], points=[(1.0, 2.0), (4.0, 6.0)], name="vg_points"
        )

        assert virtual.factors == pytest.approx([0.6, 0.8])
        assert virtual.offsets == pytest.approx([1.0, 2.0])

    def test_points_require_exactly_two_gates(self, instrument):
        inst = instrument("a", "b", "c")

        with pytest.raises(ValueError, match="exactly two gates"):
            VirtualGate(
                [inst.a, inst.b, inst.c],
                points=[(0.0, 0.0), (1.0, 1.0)],
                name="vg_points_three",
            )

    def test_identical_points_are_rejected(self, gates):
        """Two identical points give no direction, only a division by zero."""
        with pytest.raises(ValueError, match="must be different"):
            VirtualGate(
                [gates.x, gates.y],
                points=[(1.0, 1.0), (1.0, 1.0)],
                name="vg_points_same",
            )

    def test_gates_must_share_a_root_instrument(self, gates, instrument):
        other = instrument("z")

        with pytest.raises(ValueError, match="same root instrument"):
            VirtualGate(
                [gates.x, other.z],
                factors=[1.0, 1.0],
                offsets=[0.0, 0.0],
                name="vg_cross_instrument",
            )

    def test_generates_a_name_and_label_from_the_gate_names(self, gates):
        virtual = VirtualGate(
            [gates.x, gates.y], factors=[1.0, 1.0], offsets=[0.0, 0.0]
        )

        assert virtual.name == "virtual_gate_x_y"
        assert virtual.label == "Virtual Gate x y"
        assert virtual.unit == gates.x.unit
        assert virtual.instrument is gates


class TestVirtualGateBehaviour:
    """What the composite does once it exists."""

    @pytest.fixture
    def virtual(self, gates):
        return VirtualGate(
            [gates.x, gates.y],
            factors=[1.0, 2.0],
            offsets=[0.1, 0.2],
            name="vg_behaviour",
        )

    def test_setting_applies_the_transformation_to_each_gate(self, gates, virtual):
        virtual(1.0)

        assert gates.x() == pytest.approx(1.1)
        assert gates.y() == pytest.approx(2.2)

    def test_setting_zero_still_applies_the_offsets(self, gates, virtual):
        """The offset is the origin of the virtual axis, not a bias to skip."""
        virtual(0.0)

        assert gates.x() == pytest.approx(0.1)
        assert gates.y() == pytest.approx(0.2)

    def test_getting_returns_every_underlying_gate(self, gates, virtual):
        gates.x(1.0)
        gates.y(2.0)

        assert virtual.get_raw() == (1.0, 2.0)

    def test_snapshot_records_the_transformation_it_applied(self, virtual):
        """
        The snapshot is the only record of what a virtual gate meant once the
        dataset is on disk, so it must carry the coefficients and not just the
        gate names.

        """
        snapshot = virtual.snapshot_base()

        assert snapshot["param_type"] == "virtual_gate_linear"
        assert set(snapshot["gates"]) == {"x", "y"}
        assert snapshot["transformation"] == {
            "factors": [1.0, 2.0],
            "offsets": [0.1, 0.2],
        }
        assert "V_virtual" in snapshot["description"]

    def test_snapshot_of_a_rotated_gate_records_the_angle(self, gates):
        virtual = VirtualGate(
            [gates.x, gates.y], rot_angle_deg=30.0, name="vg_snapshot_rot"
        )

        snapshot = virtual.snapshot_base()

        assert snapshot["rot_angle_deg"] == 30.0
        assert snapshot["points"] is None
        # numpy scalars are not JSON-serialisable; the snapshot must cast.
        assert all(isinstance(f, float) for f in snapshot["transformation"]["factors"])

    def test_snapshot_of_a_point_defined_gate_records_the_points(self, gates):
        points = [(0.0, 0.0), (1.0, 1.0)]
        virtual = VirtualGate([gates.x, gates.y], points=points, name="vg_snapshot_pts")

        snapshot = virtual.snapshot_base()

        assert snapshot["points"] == points
        assert snapshot["rot_angle_deg"] is None


class TestMultiChannelParameter:
    """One value driven onto several channels at once."""

    def test_requires_at_least_one_channel(self):
        with pytest.raises(ValueError, match="At least one channel"):
            MultiChannelParameter([], name="mc_empty")

    def test_channels_must_share_a_root_instrument(self, gates, instrument):
        other = instrument("z")

        with pytest.raises(ValueError, match="same root instrument"):
            MultiChannelParameter([gates.x, other.z], name="mc_cross_instrument")

    def test_generates_a_name_and_label_from_the_channels(self, gates):
        multi = MultiChannelParameter([gates.x, gates.y])

        assert multi.name == "multi_channel_parameter_x_y"
        assert multi.label.startswith("MultiChannelParameter:")
        assert multi.unit == gates.x.unit
        assert multi.param_type == "gates"

    def test_setting_drives_every_channel(self, gates):
        multi = MultiChannelParameter([gates.x, gates.y], name="mc_set")

        multi(0.7)

        assert gates.x() == pytest.approx(0.7)
        assert gates.y() == pytest.approx(0.7)

    def test_getting_returns_the_common_value(self, gates):
        multi = MultiChannelParameter([gates.x, gates.y], name="mc_get")
        multi(0.7)

        assert multi.get_raw() == pytest.approx(0.7)

    def test_getting_returns_none_when_the_channels_disagree(self, gates):
        """
        A channel moved behind the composite's back means it no longer stands
        for one value, and None says so rather than reporting one channel's
        reading as if it were all of them.

        """
        multi = MultiChannelParameter([gates.x, gates.y], name="mc_mixed")
        multi(0.7)
        gates.y(0.9)

        assert multi.get_raw() is None

    def test_accepts_a_single_channel(self, gates):
        multi = MultiChannelParameter([gates.x], name="mc_single")
        multi(0.3)

        assert multi.get_raw() == pytest.approx(0.3)

    def test_accepts_any_sequence_of_channels(self, gates):
        """``param`` is typed as a Sequence, and a tuple is one."""
        multi = MultiChannelParameter((gates.x, gates.y), name="mc_tuple")

        assert multi.channels == [gates.x, gates.y]

    def test_snapshot_lists_every_channel(self, gates):
        multi = MultiChannelParameter([gates.x, gates.y], name="mc_snapshot")

        snapshot = multi.snapshot_base()

        assert snapshot["param_type"] == "gates"
        assert [channel["name"] for channel in snapshot["channels"]] == [
            gates.x.full_name,
            gates.y.full_name,
        ]
        assert all(channel["unit"] == "V" for channel in snapshot["channels"])


def test_virtual_gate_of_a_rotation_moves_along_the_expected_direction(gates):
    """
    End to end: a 45-degree virtual gate must move both gates equally, which
    is the property a diagonal detuning sweep depends on.

    """
    virtual = VirtualGate([gates.x, gates.y], rot_angle_deg=45.0, name="vg_diagonal")

    virtual(np.sqrt(2.0))

    assert gates.x() == pytest.approx(1.0)
    assert gates.y() == pytest.approx(1.0)
