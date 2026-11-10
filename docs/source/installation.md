---
myst:
  html_meta:
    description: "Install Qanary from PyPI with uv or pip. The guide also covers development installs, optional extras and migration from QCUtils."
og:description: "Install Qanary from PyPI with uv or pip. The guide also covers development installs, optional extras and migration from QCUtils."
---
# Installation

Qanary officially supports Python 3.14. Install the latest release from PyPI:

```console
uv add "qanary[drivers]"
```

```console
pip install "qanary[drivers]"
```

The `drivers` extra adds the instrument drivers
([`qcdrivers`](https://gitlab.com/squad-lab/qcdrivers) and QCoDeS contrib
drivers, which bring in the Zurich Instruments and VISA packages) that the
examples and tutorials use. Leave it out if you bring your own drivers:
`uv add qanary` installs QCoDeS, xarray, Zarr and live plotting through
[`Qimchi-Connect`](https://gitlab.com/squad-lab/qimchi-connect).

## Migrating from QCUtils

Qanary is the renamed successor to QCUtils. Replace the `qcutils` dependency
with `qanary`, change imports from `qcutils...` to `qanary...`, and rename
`QCUTILS_HOME` to `QANARY_HOME`. The default application directory has likewise
changed from `~/.qcutils` to `~/.qanary`. QCUtils installed the instrument
drivers by default; with Qanary, ask for them with the `drivers` extra.

## Development installation

To work on Qanary itself, clone the repository and install its development
and test tools:

```bash
git clone https://gitlab.com/squad-lab/qanary.git
cd qanary
uv sync --extra dev
uv run pre-commit install
```

## Optional extras

| Extra     | Installs                              | Use it for                         |
| --------- | ------------------------------------- | ---------------------------------- |
| `drivers` | `qcdrivers`, `qcodes-contrib-drivers` | Running the examples and tutorials |
| `test`    | `pytest`, `pytest-cov`, the drivers   | Running the test suite             |
| `dev`     | pre-commit, Ruff, test tools, drivers | Contributing to Qanary             |

```bash
uv sync --extra test
uv run pytest
```

The documentation dependencies live in a dependency group rather than an extra:

```bash
uv sync --group docs
uv run sphinx-build -M html docs/source docs/build
```
