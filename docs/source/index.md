---
title: Qanary
---
# Measurement Utilities for QCoDeS

**Qanary** simplifies measurement workflows and replaces QCoDeS's own
measurement layer. You declare the parameters to sweep and the quantities to
record; Qanary walks the sweep, stores the result as an
[`xarray.Dataset`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html),
and publishes it for live visualization and analysis to [Qimchi](https://gitlab.com/squad-lab/qimchi)
while it runs.

# Why Qanary?

Measurement code should describe the *experiment*, not the bookkeeping. Qanary
takes over dataset creation, incremental persistence, live publication and
metadata capture, so a sweep becomes a handful of declarative lines rather than a
nested loop with save logic threaded through it.

# Highlights

- **Declarative sweeps**: nest `Sweep` objects; the stepper walks them and fills
  a pre-allocated dataset.
- **Buffered acquisition**: hand a block of points to the instrument and read it
  back in one go, instead of a round trip per point. Much faster, where supported.
- **Live visualization**: Qanary publishes each measurement to Qimchi as it
  runs through [`qimchi-connect`](https://gitlab.com/squad-lab/qimchi-connect).
  Other measurement libraries and custom acquisition scripts can use its
  `live_measurement` context manager to publish their own live data.
- **xarray datasets**: completed measurements are stored as netCDF files and
  load directly as `xarray.Dataset` objects.
- **Lab-tested**: built for and used by [SQUAD Lab](https://squad-lab.org) at
  Forschungszentrum Jülich, Germany.

# Get Started

Qanary is a Python package managed with [uv](https://docs.astral.sh/uv/). See
[Installation](installation.md) to set up a measurement project, then
[Tutorials](tutorials/tutorials_index.md) for a first sweep.

```{toctree}
:caption: "Contents"
:maxdepth: 2

Home <self>
installation.md
tutorials/tutorials_index.md
api/index
contributing.md
development.md
changelog.md
```
