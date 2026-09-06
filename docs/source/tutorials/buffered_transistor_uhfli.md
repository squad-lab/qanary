# Transistor curves with the UHFLI

A worked single-gate measurement: sweep one DAC channel across a transistor's
gate and record the lock-in response. It is the same tree as
{doc}`buffered_basel_uhfli`, narrowed to one gate and tuned for timing.

The part worth copying is the sample-rate setup. The lock-in's demodulator rate
has to keep up with the sweep, so the example derives both from the per-point
delay:

```python
min_delay = 0.03e-3

lockin.frequency(2 * 1 / min_delay)
sample_rate(2 * 1 / min_delay)
```

Undersampling here does not fail loudly -- it quietly smears the curve -- so it
is worth setting deliberately rather than inheriting whatever the instrument was
last left at.

Full example: [`buffered_basel_transistor_test_uhfli.py`](https://gitlab.com/squad-lab/qcutils/-/blob/main/examples/buffered_basel_transistor_test_uhfli.py)
