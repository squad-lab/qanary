# QDAC-II with an MFLI, against mock instruments

Structurally the same as the other QDAC-II measurement, but the example also
builds QCoDeS `DummyInstrument` and `DummyInstrumentWithMeasurement` stand-ins
alongside the real ones. That makes it the easiest place to see the tree shape
without a rack in front of you.

```python
from qcdrivers.buffered.qdevil import NodeQDAC2
from qcdrivers.buffered.zurich import NodeMFLI

buffered_sweep = {
    "instrument": NodeQDAC2(inst=dac),
    "sweeps": [gate_sweep1, gate_sweep2],
    "nodes": [
        {
            "instrument": NodeMFLI(inst=mfli1),
            "dependent": [mfli_r, mfli_p],
        }
    ],
}
```

Full example: [`qdac_dmm_mfli_example.py`](https://gitlab.com/squad-lab/qcutils/-/blob/main/examples/qdac_dmm_mfli_example.py)
