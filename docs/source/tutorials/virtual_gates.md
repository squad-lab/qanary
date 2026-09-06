# Virtual gates

A `VirtualGate` maps one sweep coordinate onto several physical gates. Use it
to follow a meaningful direction through gate space—for example, to compensate
cross-capacitance—without calculating every physical voltage in the
measurement script.

The example uses `ShellInstrument` channels and therefore runs without
hardware.

## Define the transformation

For a virtual value $v$, each physical gate receives
$\text{factor} \times v + \text{offset}$:

```python
from qcdrivers.squad import ShellInstrument
from qanary.parameters import VirtualGate

vg = VirtualGate(
    gates=[dummy.ch1, dummy.ch2],
    factors=[1.0, -0.5],
    offsets=[0.0, 0.0],
    name="virtual_gate",
    label="Virtual gate",
)

vg(1.0)  # ch1 -> 1.0 V, ch2 -> -0.5 V
```

All gates must share one root instrument. Instead of supplying `factors` and
`offsets` directly, a two-gate virtual axis can be defined by either
`rot_angle_deg` or two `points`. The first point becomes the offset and the
direction towards the second point defines the factors.

Sweep `vg` as you would any QCoDeS parameter. Qanary records the virtual
coordinate in the dataset, while the parameter snapshot preserves the complete
physical transformation. The example plots several transformations so their
directions in gate space are easy to compare.

Full example: [`virtual_gate_parameter_example.py`](https://gitlab.com/squad-lab/qanary/-/blob/main/examples/virtual_gate_parameter_example.py)
