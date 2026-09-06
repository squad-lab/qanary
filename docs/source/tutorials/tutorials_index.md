# Tutorials

Each page walks through one of the [shipped
examples](https://gitlab.com/squad-lab/qcutils/-/tree/main/examples), explains
what it demonstrates, and links to the full script.

The two parameter tutorials run without hardware. The buffered ones describe
real instruments, so read them for the tree shape and the timing notes rather
than expecting to execute them as-is.

## Parameters

```{toctree}
:maxdepth: 1

multi_channel_parameters
virtual_gates
```

## Buffered measurements

Instrument-internal sweeps, where the hardware acquires a block of points
without a round trip per point. See {doc}`../api/buffered/buffered_index` for
the sweep-tree contract these all build on.

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
