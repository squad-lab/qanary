# Basel DAC with a UHFLI lock-in

The DAC sweeps and triggers; the lock-in acquires. This is the shape most
buffered measurements take -- a sweeping root node with acquiring children.

```python
from qcdrivers.buffered.basel import NodeBaselDAC
from qcdrivers.buffered.zurich import NodeUHFLI

buffered_sweep = {
    "instrument": NodeBaselDAC(inst=dac),
    "sweeps": [gate_sweep],
    "nodes": [
        {
            "instrument": NodeUHFLI(inst=lockin),
            "dependent": [lockin_r, lockin_p],
            "grid_mode": "exact",
            "input_trigger": get_trigger_channel(gate),
            "trigger_level": 0.3,
            "trigger_delay": 2e-6,
        }
    ],
}

measure.run([buffered_sweep], dependents=[], **run_dict)
```

`"dependent"` lists what the lock-in records -- here R and phase.
`"input_trigger"` names the physical trigger line the DAC drives, and
`grid_mode="exact"` asks for one sample per setpoint rather than a resampled
grid. The example covers both a one-dimensional sweep and a two-dimensional one
over two gates.

Full example: [`buffered_basel_with_uhfli.py`](https://gitlab.com/squad-lab/qcutils/-/blob/main/examples/buffered_basel_with_uhfli.py)
