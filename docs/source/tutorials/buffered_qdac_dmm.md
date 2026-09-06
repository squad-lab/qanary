# QDAC-II with Keysight DMMs

The QDAC-II ramps two gates and triggers two Keysight 34461A multimeters, each
recording its own voltage. It shows a tree with several sibling acquiring nodes
under one sweeping root.

```python
from qcdrivers.buffered.keysight import NodeKeysightDMM
from qcdrivers.buffered.qdevil import NodeQDAC2
from qcdrivers.keysight.dmms import Keysight34461A

buffered_sweep_dmm = {
    "instrument": NodeQDAC2(inst=dac),
    "sweeps": [gate_sweep1, gate_sweep2],
    "nodes": [
        {"instrument": NodeKeysightDMM(inst=dmm1), "dependent": [dmm_v1]},
        {"instrument": NodeKeysightDMM(inst=dmm2), "dependent": [dmm_v2]},
    ],
}
```

:::{note}
Use QCDrivers' `Keysight34461A` rather than the stock QCoDeS driver. The 34410A
accepts a sample time for ramped measurements and the 34461A does not, so the
QCDrivers version drops the timing features that would otherwise raise on the
34461A.
:::

Full example: [`buffered_qdac_with_keysight_dmm.py`](https://gitlab.com/squad-lab/qcutils/-/blob/main/examples/buffered_qdac_with_keysight_dmm.py)
