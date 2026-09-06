# Time traces

To record against time rather than a swept parameter, sweep a clock. `NodeDelay`
wraps a `Delay` instrument and acts as the root node, holding the acquisition
window open while the lock-ins sample.

```python
from qcdrivers.buffered.squad import NodeDelay
from qcdrivers.buffered.zurich import NodeMFLI, NodeUHFLI

time_sweep = Sweep(clock.time, 0, target_duration, num=n_samples)

buffered_sweep = {
    "instrument": NodeDelay(inst=clock),
    "sweeps": [time_sweep],
    "nodes": [
        {
            "instrument": NodeUHFLI(inst=lockin_RF),
            "dependent": [lockin_RF_r1, lockin_RF_p1],
        }
    ],
}
```

The example does this for a UHFLI and an MFLI, taking a target duration and a
sample count and deriving the per-point delay from them.

Full example: [`time_trace_lockin.py`](https://gitlab.com/squad-lab/qcutils/-/blob/main/examples/time_trace_lockin.py)
