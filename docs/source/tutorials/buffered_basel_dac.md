# A buffered sweep on the Basel DAC

The simplest buffered measurement: the Basel LNHR DAC plays a waveform from its
own AWG, so the sweep runs at the instrument's timing rather than one Python
round trip per point.

```python
from qcdrivers.buffered.basel import NodeBaselDAC

gate1_sweep = Sweep(gate1, 0, 1, num=5, start_delay=0, delay=0.03e-3)

buffered_sweep = {
    "instrument": NodeBaselDAC(inst=dac),
    "sweeps": [gate1_sweep],
}

measure.run([buffered_sweep], dependents=[], **run_dict)
```

Adding a second `Sweep` to `"sweeps"` makes it two-dimensional; the node
programs the outer and inner axes onto separate AWGs.

:::{note}
The two gates must sit on separate DAC boards (one high, one low), wired so each
board's sync output triggers the other. Enable high bandwidth on the channels
being swept, or the small `delay` values will not be met.
:::

Full example: [`basel_dac_sweep.py`](https://gitlab.com/squad-lab/qcutils/-/blob/main/examples/basel_dac_sweep.py)
