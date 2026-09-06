# Installation

QCUtils requires Python 3.13 and [uv](https://docs.astral.sh/uv/). uv is the
only supported installer.

QCUtils is not published on PyPI. Install it from GitLab:

```console
uv add git+https://gitlab.com/squad-lab/qcutils.git
```

Or declare it in a project's `pyproject.toml`:

```toml
[project]
dependencies = ["qcutils"]

[tool.uv.sources]
qcutils = { git = "https://gitlab.com/squad-lab/qcutils.git" }
```

```console
uv sync
```

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

