# Acquire a Basel DAC sweep with an MFLI

In this measurement, the Basel DAC generates the sweep and its synchronization
output triggers an MFLI. The sweep tree mirrors that signal flow: the DAC is the
root and the lock-in is its acquiring child.

## Build the sweep tree

```python
from qcdrivers.buffered.basel import NodeBaselDAC
from qcdrivers.buffered.zurich import NodeMFLI

buffered_sweep = {
    "instrument": NodeBaselDAC(inst=dac),
    "sweeps": [gate_sweep],
    "nodes": [
        {
            "instrument": NodeMFLI(inst=lockin),
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

The child node settings describe the acquisition:

- `dependent` lists the measured parameters, here amplitude and phase.
- `input_trigger` selects the MFLI trigger input connected to the DAC board.
- `grid_mode="exact"` requests one acquired sample for every sweep point.
- `trigger_delay` compensates the delay between the DAC update and trigger.

Set the demodulator sample rate high enough for the requested point delay, and
close any LabOne DAQ module before starting the run. A stale DAQ session can
prevent Qanary from fetching the new acquisition.

For a two-dimensional scan, both sweeps must use the same point delay. Put the
inner sweep last in `"sweeps"` and select the trigger input driven by that
inner-axis DAC board.

Full example: [`buffered_basel_with_mfli.py`](https://gitlab.com/squad-lab/qanary/-/blob/main/examples/buffered_basel_with_mfli.py)
