# %%

import matplotlib.pyplot as plt
import numpy as np
from qcdrivers.squad import ShellInstrument

from qanary.measure import Station
from qanary.parameters import VirtualGate

# %%

dummy = ShellInstrument("dummy", {})
dummy2 = ShellInstrument("dummy2", {})

# %%

v1 = None
v2 = None
v3 = None

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

dummy2.add_parameter(
    "ch3",
    label="Channel 3",
    unit="V",
    set_cmd=lambda value: globals().update(v3=value),
    get_cmd=lambda: v3,
)

# %%

st = Station("test")
st.instruments = [dummy, dummy2]

# %%

dummy_ch1 = st.add_parameter("ch1", "Channel 1", dummy.ch1)
dummy_ch2 = st.add_parameter("ch2", "Channel 2", dummy.ch2)

dummy_chx = st.add_parameter("ch1", "Channel 1", dummy.ch1, override=True)

# %%

# rotation by a certain angle, an offset by certain coordinates
theta = np.deg2rad(30)

vg_parallel = VirtualGate(
    gates=[dummy_ch1, dummy_ch2],
    name="vg_parallel",
    factors=[np.cos(theta), np.sin(theta)],
    offsets=[1, 1],
)

vg_perpendicular = VirtualGate(
    gates=[dummy_ch1, dummy_ch2],
    name="vg_perpendicular",
    factors=[-np.sin(theta), np.cos(theta)],
    offsets=[1, 1],
)

# %%

x_par, y_par = [], []
x_perp, y_perp = [], []

for i in np.linspace(0, 10, 11):
    vg_parallel.set(i)
    x_par.append(dummy_ch1())
    y_par.append(dummy_ch2())

    vg_perpendicular.set(i)
    x_perp.append(dummy_ch1())
    y_perp.append(dummy_ch2())

fig, ax = plt.subplots()

ax.plot(x_par, y_par, "o-")
ax.plot(x_perp, y_perp, "o-")

ax.set_aspect("equal", adjustable="box")
ax.grid()

# plt.show()

print(vg_parallel.snapshot())
print(vg_parallel.instrument)

print(vg_parallel.get())
print(vg_parallel.name)
print(vg_parallel.label)


def virtual_gate_2d(
    gates,
    P1: tuple[float, float],
    P2: tuple[float, float],
    name: str,
    label: str = None,
):
    # set P1 = (0, 0) for just a rotation and P2 for the direction

    Vg = VirtualGate(
        gates=gates,
        points=[P1, P2],
        name=name,
        label=label,
    )

    virtual_gate = st.add_parameter(Vg.name, Vg.label, Vg)

    return virtual_gate


Vg = virtual_gate_2d(
    gates=[dummy_ch1, dummy_ch2],
    P1=(1, 1),
    P2=(2, 3),
    name="vg_2d",
    label="Virtual Gate 2D",
)

print(Vg.snapshot())
print(Vg.instrument)

fig, ax = plt.subplots()

x, y = [], []

for i in np.linspace(-10, 10, 21):
    Vg.set(i)
    x.append(dummy_ch1())
    y.append(dummy_ch2())

print(Vg.snapshot())
print(Vg.instrument)

ax.plot(x, y, "o-")

ax.set_aspect("equal", adjustable="box")
ax.grid()

plt.show()
