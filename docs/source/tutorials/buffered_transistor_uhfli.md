# Tune a transistor sweep with a UHFLI

This example sweeps a transistor gate with the Basel DAC and acquires amplitude
and phase from two UHFLI demodulators. It implements the same measurement in
stepped and buffered forms, making the stepped result a useful reference while
the hardware timing is tuned.

## Configure every demodulator

Each selected demodulator must sample fast enough for the shortest DAC point
delay. The example initializes both rates and the excitation frequency from
that delay:

```python
min_delay = 0.03e-3

for demod in [0, 1]:
    sample_rate(2 / min_delay, demod_number=demod)

lockin.frequency1(2 / min_delay)
```

These are initial values. Verify the acquired point count and compare the curve
with a stepped run before increasing the sweep speed.

## Tune the buffered run

The buffered function makes three timing controls explicit:

- `trigger_delay` shifts acquisition relative to the DAC update.
- `tc_factor` allows for demodulator settling.
- `grid_mode="linear"` resamples the acquired values onto the sweep grid.

The example scans several trigger delays and time-constant factors. Use a small,
safe gate range for that calibration, then retain the settings that reproduce
the stepped reference without missing samples.

Full example: [`buffered_basel_transistor_test_uhfli.py`](https://gitlab.com/squad-lab/qanary/-/blob/main/examples/buffered_basel_transistor_test_uhfli.py)
