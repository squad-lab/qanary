# UHFLI sweeper-module sweeps

The UHFLI can sweep one of its own parameters with its LabOne Sweeper module.
Nothing external drives it, so the tree is a single node with no children.

```python
from qcdrivers.buffered.zurich import NodeUHFLI

f_sweep = Sweep(lockin.frequency1, 1e5, 1e6, num=501, start_delay=0, delay=0)

buffered_sweep = {
    "instrument": NodeUHFLI(inst=lockin),
    "sweeps": [f_sweep],
    "dependent": [lockin_r1, lockin_p1],
}

measure.run([buffered_sweep], dependents=[], **run_dict)
```

The example sweeps frequency and drive amplitude, on one and on two demodulator
channels.

:::{note}
The Sweeper module handles one dimension only, and every swept parameter must
map to a LabOne node. A parameter without one is rejected when the sweep is
registered.
:::

Full example: [`buffered_uhfli_sweeps.py`](https://gitlab.com/squad-lab/qcutils/-/blob/main/examples/buffered_uhfli_sweeps.py)
