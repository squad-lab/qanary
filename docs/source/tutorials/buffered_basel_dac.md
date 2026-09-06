# Run a buffered sweep with the Basel DAC

The Basel LNHR DAC can play a sweep from its internal AWGs. Once programmed,
the DAC advances at hardware speed instead of waiting for Python at every
setpoint. This is the simplest example of a buffered sweep tree: one root node
generates the sweep.

## Describe the sweep

```python
from qcdrivers.buffered.basel import NodeBaselDAC

gate1_sweep = Sweep(gate1, 0, 1, num=5, start_delay=0, delay=0.03e-3)

buffered_sweep = {
    "instrument": NodeBaselDAC(inst=dac),
    "sweeps": [gate1_sweep],
}

measure.run([buffered_sweep], dependents=[], **run_dict)
```

`NodeBaselDAC` translates the Qanary `Sweep` into an AWG waveform. The
`delay` is the requested interval between points; `start_delay` is not used by
this buffered driver.

For a two-dimensional scan, add a second `Sweep` to `"sweeps"`. The first
sweep is the outer axis and the second is the inner axis. The node assigns the
two axes to separate AWGs and returns data with the corresponding two-dimensional
shape.

:::{note}
For a two-dimensional scan, place the gates on separate DAC boards—one low and
one high—and cross-connect their synchronization signals. Enable high bandwidth
on the swept channels before requesting short point delays.
:::

Add acquiring instruments as child nodes when moving from this timing check to
a real measurement; the MFLI and UHFLI tutorials show that pattern.

Full example: [`basel_dac_sweep.py`](https://gitlab.com/squad-lab/qanary/-/blob/main/examples/basel_dac_sweep.py)
