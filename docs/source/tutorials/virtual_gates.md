# Virtual gates

A `VirtualGate` is a linear combination of real gates. Setting it distributes
the value across its constituents by a coefficient matrix, which is how you
compensate cross-capacitance and sweep along an axis that means something
physically rather than one that happens to match the wiring.

Like the multi-channel example, this one runs without hardware.

```python
from qcdrivers.squad import ShellInstrument
from qcutils.parameters import VirtualGate

vg = VirtualGate(
    name="virtual_gate",
    label="Virtual gate",
    unit="V",
    parameters=[dummy.ch1, dummy.ch2],
    coefficients=[1.0, -0.5],
)

vg(1.0)   # ch1 -> 1.0 V, ch2 -> -0.5 V
```

Sweeping the virtual gate sweeps the combination, and the dataset records the
virtual axis. The example plots the resulting gate trajectories with matplotlib
so the effect of the coefficients is visible.

Full example: [`virtual_gate_parameter_example.py`](https://gitlab.com/squad-lab/qcutils/-/blob/main/examples/virtual_gate_parameter_example.py)
