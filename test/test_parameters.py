
import matplotlib.pyplot as plt
import numpy as np

from qcutils.parameters import MultiChannelParameter
from qcutils.parameters import VirtualGate

from drivers.squad.helpers.helpers import ShellInstrument


if __name__ == "__main__":
    dummy = ShellInstrument("dummy", {})

    v1 = None
    v2 = None

    dummy.add_parameter(
        "ch1",
        label="Channel 1",
        unit="V",
        set_cmd=lambda value: globals().update(v1=value),
        get_cmd=lambda: v1,
    )

    dummy.add_parameter(
        "ch2",
        label="Channel 2",
        unit="V",
        set_cmd=lambda value: globals().update(v2=value),
        get_cmd=lambda: v2,
    )

    multi_param = MultiChannelParameter(
        channels=[dummy.ch1, dummy.ch2],
    )

    # test outputs
    multi_param.set(5.0)
    print(f"ch1: {dummy.ch1()}, ch2: {dummy.ch2()}")
    print(f"multi_param: {multi_param.get()}")

    print(multi_param.name)
    print(multi_param.label)

    print(multi_param.snapshot())




    # virtual gate test
    print("\nVirtual Gate Test:")

    #rotation by a certain angle, an offset by certain coordinates
    theta = np.deg2rad(30)

    vg_parallel = VirtualGate(
        gates=[dummy.ch1, dummy.ch2],
        factors=[np.cos(theta), np.sin(theta)],
        offsets=[1, 1],
        unit="V",
    )

    vg_perpendicular = VirtualGate(
        gates=[dummy.ch1, dummy.ch2],
        factors=[-np.sin(theta), np.cos(theta)],
        offsets=[1, 1],
        unit="V",
    )

    x_par, y_par = [], []
    x_perp, y_perp = [], []

    for i in np.linspace(0, 10, 11):
        vg_parallel.set(i)
        x_par.append(dummy.ch1())
        y_par.append(dummy.ch2())

        vg_perpendicular.set(i)
        x_perp.append(dummy.ch1())
        y_perp.append(dummy.ch2())

    fig, ax = plt.subplots()

    ax.plot(x_par, y_par, "o-")
    ax.plot(x_perp, y_perp, "o-")

    ax.set_aspect('equal', adjustable='box')
    ax.grid()

    plt.show()

    print(vg_parallel.snapshot())

    print(vg_parallel.get())
    print(vg_parallel.name)
    print(vg_parallel.label)
