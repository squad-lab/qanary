# Record buffered time traces

A time trace has no physical sweep generator, but it still needs a root node to
define the acquisition window. Qanary represents that window as a sweep over a
clock parameter. `NodeDelay` wraps the clock's `Delay` instrument and triggers
the lock-in acquisition beneath it.

## Match the sweep to the sample rate

Choose a target duration and number of samples, configure the lock-in, and read
back its actual sample rate. Derive the point delay from that returned value;
the instrument may quantize the requested rate.

```python
target_sample_rate = n_samples / target_duration
lockin_DC.sample_rate(target_sample_rate)

actual_sample_rate = lockin_DC.sample_rate()
point_delay = 1 / actual_sample_rate
```

## Build the time-trace tree

```python
from qcdrivers.buffered.squad import NodeDelay
from qcdrivers.buffered.zurich import NodeMFLI

time_sweep = Sweep(
    clock.time,
    point_delay,
    n_samples * point_delay,
    num=n_samples,
    start_delay=0,
    delay=point_delay,
)

buffered_sweep = {
    "instrument": NodeDelay(inst=clock),
    "sweeps": [time_sweep],
    "nodes": [
        {
            "instrument": NodeMFLI(inst=lockin_DC),
            "dependent": [lockin_DC_r, lockin_DC_p],
            "grid_mode": "exact",
            "force_trigger": True,
        }
    ],
}
```

`force_trigger=True` starts the acquisition without an external trigger, while
`grid_mode="exact"` preserves the requested sample count. The example provides
equivalent buffered functions for a UHFLI and an MFLI, plus a stepped version
for comparison. Prefer the buffered form for long traces because Python does
not have to schedule every sample individually.

Full example: [`time_trace_lockin.py`](https://gitlab.com/squad-lab/qanary/-/blob/main/examples/time_trace_lockin.py)
