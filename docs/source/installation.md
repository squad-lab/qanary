# Installation

QCUtils requires Python 3.13. Install the latest release from PyPI:

```console
uv add qcutils
```

```console
pip install qcutils
```

QCUtils pulls in its QCoDeS, Zurich Instruments, live-visualization
([`qimchi-connect`](https://gitlab.com/squad-lab/qimchi-connect)) and instrument
driver ([`qcdrivers`](https://gitlab.com/squad-lab/qcdrivers)) dependencies, so
there is no need to add them separately.

## Development installation

To work on QCUtils itself, clone the repository and install its development
and test tools:

```bash
git clone https://gitlab.com/squad-lab/qcutils.git
cd qcutils
uv sync --extra dev
uv run pre-commit install
```

## Optional extras

| Extra  | Installs                        | Use it for              |
| ------ | ------------------------------- | ----------------------- |
| `test` | `pytest`, `pytest-cov`          | Running the test suite  |
| `dev`  | pre-commit, Ruff, and test tools | Contributing to QCUtils |

```bash
uv sync --extra test
uv run pytest
```

The documentation dependencies live in a dependency group rather than an extra:

```bash
uv sync --group docs
uv run sphinx-build -M html docs/source docs/build
```

