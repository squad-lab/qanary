# Contributing to Qanary

Thank you for contributing to Qanary. Qanary provides a measurement layer on
top of QCoDeS: it runs parameter sweeps, stores results as an `xarray.Dataset`,
and publishes live data to [Qimchi](https://gitlab.com/squad-lab/qimchi). Contributions may include
measurement features, support for buffered acquisition on new instruments, bug
fixes, tests, tutorials, and documentation.

## Development

Qanary uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Python support policy

We support the second-newest stable Python feature release. This balances timely
security and bug-fix support with enough time for our dependencies to stabilize
on each new Python version.

### Setup

Run this once per clone:

```console
git clone https://gitlab.com/squad-lab/qanary.git
cd qanary
uv sync --extra dev && uv run pre-commit install
```

Work on a branch of your own, one per change.

### Before you push

```console
uv run pytest
uv run pre-commit run --all-files
```

The first command runs the tests. The second applies Ruff's lint fixes and
formatter across the repository.

### Conventions

Merge requests should follow these conventions:

- **Public and internal API.** Every public module declares `__all__`; anything
  else carries a leading underscore and is internal, even where one module
  imports it from another.
- **Docstrings.** Google-style, with typed `Args:`, `Returns:` and `Raises:`
  sections where useful. Tests do not need docstrings if self-explanatory; otherwise, please add them.
- **Docs build.** The API reference is generated from those docstrings, so a new
  public callable needs no separate API page. Build the docs after substantial
  docstring changes.

  ```console
  uv sync --extra dev --group docs
  uv run sphinx-build -M html docs/source docs/build -W --keep-going
  ```

  Sync both the extra and the group.

The [Development](https://qanary.squad-lab.org/development.html) page explains
these conventions with examples. Read it
before making changes.

## Test strategy

The pytest suite in `tests/` covers sweeps, parameters, measurements, storage, checkpoints,
logging, and the live registry. CI runs Python 3.13 and 3.14 on Linux, plus
Python 3.14 on Windows for platform-specific storage and file-locking behavior.
All jobs must pass before merging.

Tests must use dummy instruments because CI has no measurement hardware. The one exception is `tests/test_examples.py`. As examples require real instruments, it only checks their
syntax and imports without actually running them.

Include a regression test with each bug fix and tests for each new feature.

## Issues

Report bugs and propose changes through the
[issue tracker](https://gitlab.com/squad-lab/qanary/-/issues). For a larger
change, please open an issue before you start, so the approach can be discussed
first.

A bug report should give the Qanary and QCoDeS versions, the instruments
involved and how they are connected, and the smallest sweep that reproduces the
problem (an MWE or Minimal Working Example). Include the traceback and mention whether Qanary wrote a dataset.

## Merge requests

Feature branches merge into `preview`, which is linted, tested, documented and
package-built without publishing. `preview` then merges into `main`, where
semantic-release creates and publishes the release.

Describe what the change does, why, and how it was tested. Keep unrelated
changes in separate merge requests, and call out anything you could not test
without hardware.

Releases are cut from the commit messages, so they follow
[Conventional Commits](https://www.conventionalcommits.org/):
- `fix:` for a patch release,
- `feat:` for a minor one, and
- a `BREAKING CHANGE:` footer for a major one.
 
Other prefixes such as `docs:`, `test:`, `chore:`, and `refactor:` do not create
a release. Use `fix:` or `feat:` for changes that should ship.

Do not edit the version in `pyproject.toml`, and do not write an Unreleased
section in `CHANGELOG.md` by hand: CI owns the version, and semantic-release
generates the changelog from the commit history.

Every merge request must pass the lint, test, security, and secret-detection
jobs in the pipeline.
