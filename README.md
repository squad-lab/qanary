# Qanary

[![pipeline](https://gitlab.com/squad-lab/qanary/badges/main/pipeline.svg?ignore_skipped=true&key_text=pipeline&key_width=60)](https://gitlab.com/squad-lab/qanary/-/pipelines?ref=main)
[![tests](https://gitlab.com/squad-lab/qanary/badges/main/pipeline.svg?job=pytest%3A%20%5B3.13%5D&ignore_skipped=true&key_text=tests&key_width=40)](https://gitlab.com/squad-lab/qanary/-/pipelines?ref=main)
[![coverage](https://gitlab.com/squad-lab/qanary/badges/main/coverage.svg?key_text=coverage&key_width=64)](https://gitlab.com/squad-lab/qanary/-/jobs)
[![latest release](https://gitlab.com/squad-lab/qanary/-/badges/release.svg?key_text=release&key_width=54)](https://gitlab.com/squad-lab/qanary/-/releases)
[![PyPI version](https://img.shields.io/pypi/v/qanary.svg)](https://pypi.org/project/qanary/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23536660.svg)](https://doi.org/10.5281/zenodo.23536660)
[![documentation](https://img.shields.io/badge/docs-qanary.squad--lab.org-blue.svg)](https://qanary.squad-lab.org/)

Qanary runs parameter sweeps, records them as xarray datasets, and publishes
live results to Qimchi.

## Installation

Qanary officially supports Python 3.14. Install the latest release from PyPI:

```console
uv add "qanary[drivers]"
```

```console
pip install "qanary[drivers]"
```

The `drivers` extra adds the instrument drivers used by the examples
([`qcdrivers`](https://gitlab.com/squad-lab/qcdrivers) and QCoDeS contrib
drivers). Leave it out if you bring your own.

## Migrating from QCUtils

Qanary is the renamed successor to QCUtils. Replace the `qcutils` dependency
with `qanary`, change imports from `qcutils...` to `qanary...`, and rename
`QCUTILS_HOME` to `QANARY_HOME`. The default application directory has likewise
changed from `~/.qcutils` to `~/.qanary`.

## Preparing a measurement project

The examples below use [`uv`](https://docs.astral.sh/uv/) to manage the project
environment and run commands.

### Initializing a Measurement

Create a project for the measurement:

```sh
uv init measurement_name
cd measurement_name
```

Add Qanary to the project with:

```sh
uv add "qanary[drivers]"
```

> [!note] NOTE  
> Qanary includes QCoDeS and live plotting. The `drivers` extra adds the
> instrument drivers, including the Zurich Instruments and VISA packages.

### Running Commands

Run modules inside the project environment with:

```sh
uv run -m package.method
```

### Using `pyproject.toml`

You can also declare Qanary directly in `pyproject.toml`:

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

Then install the declared dependencies:

```sh
uv sync
```

## Measurement

For complete measurement scripts, see the [examples](https://gitlab.com/squad-lab/qanary/-/tree/main/examples) directory.

### Live and disk formats

`qanary` uses a split data path:

- Live visualization: in-memory Zarr snapshots published through
  [Qimchi-Connect](https://gitlab.com/squad-lab/qimchi-connect), with a
  temporary disk checkpoint for recovery
- Completed measurement: a netCDF (`.nc`) file in the configured data directory

Qimchi discovers a Qanary measurement automatically while it runs. After a
successful final export, Qanary removes the temporary Zarr checkpoint and the
netCDF file remains as the measurement record.

## Development

To contribute, clone the repository:
```sh
git clone https://gitlab.com/squad-lab/qanary
```
Navigate into the cloned directory and install the package in development mode:
```sh
cd qanary
uv sync --extra dev
uv run pre-commit install
```

## Citing Qanary

Please cite Qanary when using it in your work:

```bibtex
@misc{qanary,
  author       = {Anupam, Spandan and Schreibing, Simon and Shivottam, Jyotirmaya and Mourik, Vincent},
  title        = {Qanary: Sweep and measurement utilities for QCoDeS},
  year         = {2026},
  note         = {Version 0.13.0, software},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.23536660},
  url          = {https://doi.org/10.5281/zenodo.23536660}
}
```

This DOI covers all archived releases. To cite a specific version, use its DOI from the
[Zenodo record](https://doi.org/10.5281/zenodo.23536660).
[CITATION.cff](CITATION.cff) has the same information in machine-readable form.

## Authors

- Spandan Anupam: [s.anupam@fz-juelich.de](mailto:s.anupam@fz-juelich.de)
- Simon Schreibing: [s.schreibing@fz-juelich.de](mailto:s.schreibing@fz-juelich.de)
- Jyotirmaya Shivottam: [shivottam@proton.me](mailto:shivottam@proton.me)
- Vincent Mourik: [v.mourik@fz-juelich.de](mailto:v.mourik@fz-juelich.de)

## License

Qanary is free software under the [AGPL-3.0-or-later](LICENSE). Dependencies are installed separately under their own licenses.

Copyright © 2024–2026 Forschungszentrum Jülich GmbH, PGI-11.
