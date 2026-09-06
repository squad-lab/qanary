# Acquire a QDAC-II sweep with Keysight DMMs

This example programs a two-dimensional QDAC-II sweep and records two voltage
signals with Keysight 34461A multimeters. It also shows how one sweep generator
can trigger several acquisition devices: both DMM nodes are children of the
same QDAC-II root.

## Build the sweep tree

```python
from qcdrivers.buffered.keysight import NodeKeysightDMM
from qcdrivers.buffered.qdevil import NodeQDAC2
from qcdrivers.keysight.dmms import Keysight34461A

buffered_sweep_dmm = {
    "instrument": NodeQDAC2(inst=dac),
    "sweeps": [gate_sweep1, gate_sweep2],
    "output_trigger": 4,
    "trigger_type": "step",
    "trigger_width": 1e-3,
    "nodes": [
        {
            "instrument": NodeKeysightDMM(inst=dmm1),
            "dependent": [dmm_v1],
            "input_trigger": 1,
        },
        {
            "instrument": NodeKeysightDMM(inst=dmm2),
            "dependent": [dmm_v2],
            "input_trigger": 1,
        },
    ],
}
```

`output_trigger` selects the QDAC-II port wired to the DMMs. Each child then
selects its corresponding `input_trigger`. With `trigger_type="step"`, the DAC
emits a pulse at every setpoint; `trigger_width` must satisfy the input
requirements of both meters.

The delays on the two `Sweep` objects describe the fastest requested ramp. The
DMM configuration may impose a slower practical limit, so start conservatively
and shorten the delay only after checking that every point is returned.

:::{note}
Use QCDrivers' `Keysight34461A` for this example. Unlike the 34410A, the 34461A
does not accept the same sample-time configuration for ramped measurements;
the QCDrivers implementation avoids sending those unsupported settings.
:::

Full example: [`buffered_qdac_with_keysight_dmm.py`](https://gitlab.com/squad-lab/qanary/-/blob/main/examples/buffered_qdac_with_keysight_dmm.py)
