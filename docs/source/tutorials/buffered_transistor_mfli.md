# Tune a transistor sweep with an MFLI

This example sweeps one Basel DAC channel across a transistor gate and records
the MFLI amplitude and phase. It includes both stepped and buffered versions so
you can compare a known-good slow measurement with the hardware-timed result.

## Start from the sweep timing

The demodulator must produce enough samples during each DAC step. The example
sets the lock-in frequency and sample rate from the shortest point delay it
plans to use:

```python
min_delay = 0.03e-3

lockin.frequency(2 / min_delay)
sample_rate(2 / min_delay)
```

Treat this as a starting point, not a universal setting. A rate that is too low
can smear or drop points; a rate that is unnecessarily high can also make the
DAQ fetch unreliable. Confirm the returned data while tuning it.

## Tune the acquisition

The buffered function exposes the settings most likely to need adjustment:

- `trigger_delay` aligns the MFLI sample window with the updated gate voltage.
- `tc_factor` controls how much settling time is allowed relative to the
  demodulator time constant.
- `grid_mode="linear"` lets the DAQ module place samples on a uniform grid.

Use the stepped sweep as a reference, then compare it with buffered runs over a
small range of trigger delays and time-constant factors. This is more reliable
than carrying timing values over from a different device or wiring setup.

Full example: [`buffered_basel_transistor_test_mfli.py`](https://gitlab.com/squad-lab/qanary/-/blob/main/examples/buffered_basel_transistor_test_mfli.py)
