# Qanary

[![pipeline](https://gitlab.com/squad-lab/qanary/badges/main/pipeline.svg?ignore_skipped=true&key_text=pipeline&key_width=60)](https://gitlab.com/squad-lab/qanary/-/pipelines?ref=main)
[![tests](https://gitlab.com/squad-lab/qanary/badges/main/pipeline.svg?job=pytest%3A%20%5B3.13%5D&ignore_skipped=true&key_text=tests&key_width=40)](https://gitlab.com/squad-lab/qanary/-/pipelines?ref=main)
[![coverage](https://gitlab.com/squad-lab/qanary/badges/main/coverage.svg?key_text=coverage&key_width=64)](https://gitlab.com/squad-lab/qanary/-/jobs)
[![latest release](https://gitlab.com/squad-lab/qanary/-/badges/release.svg?key_text=release&key_width=54)](https://gitlab.com/squad-lab/qanary/-/releases)
[![PyPI version](https://img.shields.io/pypi/v/qanary.svg)](https://pypi.org/project/qanary/)

Qanary simplifies measurement workflows and replaces QCoDeS for handling measurements.

## Installation

Qanary requires Python 3.13. Install the latest release from PyPI:

```console
uv add qanary
```

```console
pip install qanary
```

## Preparing a measurement project

We manage and run our measurements using the tool `uv`. To get started, follow the official [Astral installation guide](https://astral.sh/uv/) to install `uv`. It handles virtual environments and package dependencies for you, simplifying Python package management.

### Initializing a Measurement

To start a measurement, first initialize a `uv` project with the desired measurement name:

```sh
uv init measurement_name
cd measurement_name
```

Add Qanary to the project with:

```sh
uv add qanary
```

> [!note] NOTE  
> Qanary installs its QCoDeS, Zurich Instruments, live-visualization, and driver dependencies, so there is no need to add them separately.

### Running Commands

Instead of manually activating a virtual environment and running `python -m package.method`, you can use `uv` to streamline this:

```sh
uv run -m package.method
```

### Using `pyproject.toml`

If you prefer to skip the `add` commands, you can set up the project directly with a `pyproject.toml` file. Here's an example:

```toml
[project]
name = "example"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "qanary",
]
```

After updating the `pyproject.toml` file, you can sync the dependencies with:

```sh
uv sync
```

This will automatically fetch and install the required packages.

## Measurement

For complete measurement scripts, see the [examples](https://gitlab.com/squad-lab/qanary/-/tree/main/examples) directory.

### Live & Disk Format

`qanary` uses a split data path:

- Live visualization: in-memory Zarr snapshots published through
  [`qimchi-connect`](https://gitlab.com/squad-lab/qimchi-connect), with a
  temporary disk checkpoint for recovery
- Completed measurement: a netCDF (`.nc`) file in the configured data directory

Qimchi discovers a Qanary measurement automatically while it runs. After a
successful final export, Qanary removes the temporary Zarr checkpoint and the
netCDF file remains as the measurement record.

## Development

We welcome contributions to `qanary`! To get started with development, clone the repository:
```sh
git clone https://gitlab.com/squad-lab/qanary
```
Navigate into the cloned directory and install the package in development mode:
```sh
cd qanary
uv sync --extra dev
uv run pre-commit install
```

## Authors

- Spandan Anupam: [s.anupam@fz-juelich.de](mailto:s.anupam@fz-juelich.de)
