"""
Tests for the Parameter arithmetic operators installed by ``qcutils``.

Importing ``qcutils`` patches ``__add__``, ``__sub__``, ``__mul__``,
``__truediv__`` and ``__pow__`` onto QCoDeS' ``Parameter``, so measurement
scripts can write ``dmm.v1 - dmm.v2`` and pass the result straight to a
measurement as a dependent.

Two properties matter and neither is obvious from the implementation. The
result is a *lazy* parameter -- it reads its operands when it is called, not
when it is built -- and the additive operators refuse mismatched units, while
the multiplicative ones combine them instead.

"""

import pytest
from qcodes.parameters import Parameter

# Importing qcutils is what installs the operators; the import is the fixture.
import qcutils  # noqa: F401


def test_addition_and_subtraction_combine_values(gates):
    """The additive operators read both operands and keep the shared unit."""
    gates.x(1.5)
    gates.y(0.5)

    total = gates.x + gates.y
    difference = gates.x - gates.y

    assert total() == pytest.approx(2.0)
    assert difference() == pytest.approx(1.0)
    assert total.unit == difference.unit == gates.x.unit


def test_multiplication_and_division_combine_units(gates):
    """Multiplicative operators derive a compound unit from their operands."""
    gates.x(3.0)
    gates.y(2.0)

    product = gates.x * gates.y
    quotient = gates.x / gates.y
    power = gates.x**gates.y

    assert product() == pytest.approx(6.0)
    assert quotient() == pytest.approx(1.5)
    assert power() == pytest.approx(9.0)
    assert product.unit == "VV"
    assert quotient.unit == "V/V"
    assert power.unit == "V^V"


def test_derived_parameters_are_named_after_their_operands(gates):
    """The generated name concatenates the operand names, as scripts expect."""
    assert (gates.x + gates.y).name == "xy"


def test_derived_parameters_read_their_operands_lazily(gates):
    """
    The result must track later changes to its operands.

    A sweep builds the derived parameter once, before the sweep starts, and
    reads it at every point. If the operands were captured at construction
    time, every point would return the same number.

    """
    gates.x(1.0)
    gates.y(1.0)
    total = gates.x + gates.y
    assert total() == pytest.approx(2.0)

    gates.x(10.0)

    assert total() == pytest.approx(11.0)


@pytest.mark.parametrize("operator", ["__add__", "__sub__"])
def test_additive_operators_reject_mismatched_units(gates, instrument, operator):
    """Adding volts to amps is a scripting error, not a silent conversion."""
    other = instrument("ch1")
    other.ch1.unit = "A"

    with pytest.raises(AssertionError, match="Units must be the same"):
        getattr(gates.x, operator)(other.ch1)


def test_multiplicative_operators_allow_mismatched_units(gates, instrument):
    """Volts times amps is watts; the compound unit records that."""
    current = instrument("ch1")
    current.ch1.unit = "A"
    gates.x(2.0)
    current.ch1(3.0)

    power = gates.x * current.ch1

    assert power() == pytest.approx(6.0)
    assert power.unit == "VA"


def test_division_by_a_zeroed_parameter_raises(gates):
    """Division is not guarded, and a silent inf would poison a dataset."""
    gates.x(1.0)
    gates.y(0.0)

    with pytest.raises(ZeroDivisionError):
        (gates.x / gates.y)()


def test_operators_are_installed_on_the_qcodes_class():
    """The patch is global: any Parameter gets the operators, not just gates."""
    for name in ("__add__", "__sub__", "__mul__", "__truediv__", "__pow__"):
        assert getattr(Parameter, name).__module__ == "qcutils"
