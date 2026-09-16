# Contributing to Qanary

Thank you for contributing to Qanary. Qanary replaces QCoDeS's measurement
layer: you declare the parameters to sweep and the quantities to record, and it
walks the sweep, stores the result as an `xarray.Dataset`, and publishes it live
to [Qimchi](https://gitlab.com/squad-lab/qimchi). Contributions may include
measurement features, support for buffered acquisition on new instruments, bug
fixes, tests, tutorials, and documentation.

## Development

Qanary requires Python 3.13 and uses [uv](https://docs.astral.sh/uv/) for
dependency management.

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

The first runs the test suite. The second applies Ruff's lint fixes and its formatter to the whole repository.

### Conventions

Qanary has a few rules that a merge request is reviewed against:

- **Public and internal API.** Every public module declares `__all__`; anything
  else carries a leading underscore and is internal, even where one module
  imports it from another.
- **Docstrings.** Google-style, with typed `Args:`, `Returns:` and `Raises:`
  sections where they apply. Tests are exempt.
- **Docs build.** The API reference is generated from those docstrings, so a new
  public callable needs no docs edit -- but build the docs after a substantial
  docstring change, because malformed ones surface only as Sphinx warnings.

  ```console
  uv sync --extra dev --group docs
  uv run sphinx-build -M html docs/source docs/build -W --keep-going
  ```

  Sync both the extra and the group.

Each of these is set out in full, with the reasoning and the worked examples, on
the [Development](https://qanary.squad-lab.org/development.html) page. Read it
before your first substantial change.

## Issues

Report bugs and propose changes through the
[issue tracker](https://gitlab.com/squad-lab/qanary/-/issues). For a larger
change, please open an issue before you start, so the approach can be discussed
first.

A bug report should give the Qanary and QCoDeS versions, the instruments
involved and how they are connected, and the smallest sweep that reproduces the
problem (a Minimal Working Example). Include the traceback, and say whether the dataset was written.

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
- `feat:` for a minor one, and,
- a `BREAKING CHANGE:` footer for a major one.
 
Anything else -- `docs:`, `test:`, `chore:`, `refactor:` -- is a valid
subject that does not lead to a release. A change that should ship needs the right prefix.

Do not edit the version in `pyproject.toml`, and do not write an Unreleased
section in `CHANGELOG.md` by hand: CI owns the version, and semantic-release
generates the changelog from the commit history.

Every merge request must pass the lint, test, security, and secret-detection
jobs in the pipeline.
