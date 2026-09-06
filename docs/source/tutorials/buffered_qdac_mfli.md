# Combine stepped and buffered sweeps

Qanary can run ordinary stepped sweeps and an instrument-buffered sweep in the
same measurement. This example steps two dummy parameters in Python while a
QDAC-II performs a two-dimensional buffered sweep and an MFLI acquires its
response.

The dummy instruments make the outer part of the example reproducible, but the
buffered branch still requires a QDAC-II and MFLI.

## Define the buffered branch

```python
from qcdrivers.buffered.qdevil import NodeQDAC2
from qcdrivers.buffered.zurich import NodeMFLI

buffered_sweep = {
    "instrument": NodeQDAC2(inst=dac),
    "sweeps": [gate_sweep1, gate_sweep2],
    "output_trigger": 1,
    "trigger_type": "step",
    "nodes": [
        {
            "instrument": NodeMFLI(inst=mfli1),
            "dependent": [mfli_r, mfli_p],
            "input_trigger": 1,
        }
    ],
}
```

The QDAC-II emits one trigger per setpoint, and the MFLI records amplitude and
phase. The trigger port numbers must match the physical connection between the
two instruments.

Pass the stepped sweeps and buffered branch together to `measure.run`:

```python
measure.run(
    [gate_sweep_dummy1, gate_sweep_dummy2, buffered_sweep],
    **run_dict,
)
```

The resulting dataset uses the two stepped parameters as outer coordinates and
the QDAC-II sweeps as inner coordinates. This pattern is useful when slow
controls surround a fast acquisition block.

Full example: [`qdac_dmm_mfli_example.py`](https://gitlab.com/squad-lab/qanary/-/blob/main/examples/qdac_dmm_mfli_example.py)
