# Acquire a Basel DAC sweep with a UHFLI

This setup uses the Basel DAC as the sweep generator and a UHFLI as the
acquisition device. The DAC synchronization output provides the hardware
trigger, so the sweep tree follows the same root-to-child direction.

## Build the sweep tree

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
            "demod_channels": [0],
            "grid_mode": "exact",
            "input_trigger": get_trigger_channel(gate),
            "trigger_level": 0.3,
            "trigger_delay": 2e-6,
        }
    ],
}

measure.run([buffered_sweep], dependents=[], **run_dict)
```

The child node settings describe what the UHFLI should acquire:

- `dependent` contains the amplitude and phase parameters to store.
- `demod_channels` identifies the demodulators behind those parameters.
- `input_trigger` selects the trigger input wired to the active DAC board.
- `grid_mode="exact"` requests one sample for every sweep point.
- `trigger_delay` aligns acquisition with the updated DAC output.

Configure the sample rate of every selected demodulator before running the
sweep. For the lab wiring used by the example, the trigger inputs should use
1 kΩ impedance. Close any open LabOne DAQ module so it does not conflict with
the acquisition started by Qanary.

For a two-dimensional scan, use the same point delay on both axes, place the
inner sweep last, and trigger from the DAC board that drives that inner axis.

Full example: [`buffered_basel_with_uhfli.py`](https://gitlab.com/squad-lab/qanary/-/blob/main/examples/buffered_basel_with_uhfli.py)
