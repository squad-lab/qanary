from qcodes.parameters import Parameter


def add(self, other: Parameter):
    """Add functionality for qcodes Parameter"""
    assert self.unit == other.unit, "Units must be the same"

    def add_func():
        return float(self.get_raw()) + float(other.get_raw())

    add_param = Parameter(
        name=f"{self.name}{other.name}", unit=self.unit, get_cmd=add_func
    )
    return add_param


def sub(self, other: Parameter):
    """Subtract functionality for qcodes Parameter"""
    assert self.unit == other.unit, "Units must be the same"

    def sub_func():
        return float(self.get_raw()) - float(other.get_raw())

    sub_param = Parameter(
        name=f"{self.name}{other.name}", unit=self.unit, get_cmd=sub_func
    )
    return sub_param


def mul(self, other: Parameter):
    """Multiply functionality for qcodes Parameter"""

    def mul_func():
        return float(self.get_raw()) * float(other.get_raw())

    mul_param = Parameter(
        name=f"{self.name}{other.name}",
        unit=f"{self.unit}{other.unit}",
        get_cmd=mul_func,
    )
    return mul_param


def truediv(self, other: Parameter):
    """Division functionality for qcodes Parameter"""

    def truediv_func():
        return float(self.get_raw()) / float(other.get_raw())

    truediv_param = Parameter(
        name=f"{self.name}{other.name}",
        unit=f"{self.unit}/{other.unit}",
        get_cmd=truediv_func,
    )
    return truediv_param


def power(self, other: Parameter):
    """Power functionality for qcodes Parameter"""

    def pow_func():
        return float(self.get_raw()) ** float(other.get_raw())

    pow_param = Parameter(
        name=f"{self.name}{other.name}",
        unit=f"{self.unit}^{other.unit}",
        get_cmd=pow_func,
    )
    return pow_param


Parameter.__add__ = add
Parameter.__sub__ = sub
Parameter.__mul__ = mul
Parameter.__truediv__ = truediv
Parameter.__pow__ = power
