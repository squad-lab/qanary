# Sweep a UHFLI with its LabOne Sweeper

The UHFLI can generate and acquire a sweep entirely through its LabOne Sweeper
module. Because no external instrument drives the setpoints, the buffered tree
contains a single node that owns both the sweep and its dependent parameters.

## Configure the node

```python
from qcdrivers.buffered.zurich import NodeUHFLI

f_sweep = Sweep(lockin.frequency1, 1e5, 1e6, num=501, start_delay=0, delay=0)

buffered_sweep = {
    "instrument": NodeUHFLI(inst=lockin),
    "sweeps": [f_sweep],
    "dependent": [lockin_r1, lockin_p1],
    "phase_unwrap": True,
    "settling_inaccuracy": 0.1e-3,
    "acquisition": "sweeper",
}

measure.run([buffered_sweep], dependents=[], **run_dict)
```

`acquisition="sweeper"` selects the LabOne module. `settling_inaccuracy`
controls how closely the demodulator must settle at each point, so tighter
values can increase the run time. `phase_unwrap=True` avoids discontinuities at
the ±180° boundary.

The example includes frequency and output-amplitude sweeps and records one or
two demodulator channels. The `delay` and `start_delay` values on the Qanary
`Sweep` are not used in this mode; settling is controlled by the LabOne
settings instead.

:::{note}
The Sweeper module supports one sweep dimension. The swept QCoDeS parameter
must also map to a LabOne node; registration fails when that mapping is absent.
:::

Full example: [`buffered_uhfli_sweeps.py`](https://gitlab.com/squad-lab/qanary/-/blob/main/examples/buffered_uhfli_sweeps.py)
