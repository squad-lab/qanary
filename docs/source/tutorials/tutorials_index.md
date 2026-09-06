# Tutorials

These tutorials introduce Qanary through the [example
scripts](https://gitlab.com/squad-lab/qanary/-/tree/main/examples). Each page
focuses on the part of an example that is useful in a real measurement: how to
model the sweep, arrange buffered instruments, and choose the relevant timing
and trigger settings.

The parameter tutorials run without hardware. The buffered tutorials use lab
instruments and are intended as starting points; update addresses, channels,
wiring, limits, and data paths before running them.

## Parameters

```{toctree}
:maxdepth: 1

multi_channel_parameters
virtual_gates
```

## Buffered measurements

Buffered measurements let the instruments generate and acquire a complete
block of points without a Python round trip at every setpoint. Start with the
{doc}`buffered_basel_dac` tutorial for a single node, then add an acquiring
instrument with {doc}`buffered_basel_mfli` or {doc}`buffered_basel_uhfli`.

See {doc}`../api/buffered/buffered_index` for the complete sweep-tree contract.

```{toctree}
:maxdepth: 1

buffered_basel_dac
buffered_basel_mfli
buffered_basel_uhfli
buffered_transistor_mfli
buffered_transistor_uhfli
buffered_uhfli_sweeper
buffered_qdac_dmm
buffered_qdac_mfli
buffered_time_traces
```
