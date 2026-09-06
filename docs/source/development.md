# Development

This page defines the conventions for contributing Python code to QCUtils. The
rules below apply to production code in `src/qcutils` and should be followed in
new code and when modifying existing code.

## Getting set up

```bash
git clone https://gitlab.com/squad-lab/qcutils.git
cd qcutils
uv sync --extra dev
uv run pre-commit install    # once per clone
```

The `dev` extra carries `pre-commit`, Ruff, and the test tools. Everyday
commands:

| Task | Command |
| ---- | ------- |
| Run the tests | `uv run pytest` |
| Tests with coverage | `uv run pytest --cov` |
| Lint (includes import sorting) | `uv run ruff check .` |
| Format | `uv run ruff format .` |
| Build the docs | `uv sync --group docs && uv run sphinx-build -M html docs/source docs/build` |

Ruff's configuration lives in `pyproject.toml`, and the pre-commit hooks use it
directly, so a local run and CI agree. Import sorting comes from ruff's `I`
rules under `ruff check` -- `ruff format` does not sort imports.

## Public and internal API

Each public implementation module declares `__all__`. Anything not listed there
is internal and is prefixed with an underscore, including across module
boundaries: `qcutils.sweep` imports `_arm_instruments`, `_fetch_results`, and
`_buffered_sweep_progress_info` from `qcutils.buffered.sweep`. They stay private
despite the cross-module import because none of them is meant for measurement
scripts. Package `__init__.py` files are the exception: the top-level package
installs QCUtils's arithmetic operators on QCoDeS parameters, while
`qcutils.buffered` only marks the subpackage.

When adding a callable, decide which side of that line it falls on. If it is
part of the measurement-script API, add it to `__all__`; otherwise prefix it.
Renaming a public name later is a breaking change, so err towards private.

## Commits and releases

Releases are cut by `semantic-release` from the commit messages, so commit
subjects follow [Conventional Commits](https://www.conventionalcommits.org):

```text
fix: continue sweeps through transient Zarr errors
feat: add segmented sweep support
refactor: remove deprecated instrument classes
```

`fix:` produces a patch release, `feat:` a minor one, and a `BREAKING CHANGE:`
footer a major one. A commit that does not follow the convention produces no
release at all, so a bump can silently fail to happen -- if a change should ship,
label it.

Do not edit the version in `pyproject.toml` or write an Unreleased section in
`CHANGELOG.md` by hand. CI owns the version, and semantic-release generates the
changelog from commit history. Keep prospective commit messages in `todo.md`
until the corresponding changes are committed.

Branch model: feature branches merge into `preview`, which is linted, tested,
documented, and package-built without publishing a release. `preview` then
merges into `main`, where semantic-release creates and publishes releases.

## Python docstrings

### Google-style sections

Production callable docstrings use `Args:`, `Returns:`, and `Raises:` sections
when those sections apply. Do not add empty sections or placeholder entries
such as `None.`. When multiple sections apply, keep them in that order. Include
explicit types so the documentation is useful when read in source code and when
rendered by Sphinx.

Use these forms:

```text
Args:
    argument_name (ArgumentType): Description.

Returns:
    ReturnType: Description.

Raises:
    ExceptionType: Condition that raises the exception.
```

For arguments:

- Add `Args:` when the callable accepts arguments other than `self` or `cls`.
- Write the argument name first and its type in parentheses.
- Use the same type expression as the function annotation where practical.
- Describe the argument's meaning, constraints, units, and defaults when they
  are relevant.

For return values:

- Add `Returns:` when the callable returns a meaningful value. Omit it when the
  callable only returns `None`.
- Begin with the complete return type.
- Describe the meaning and structure of the returned value, not merely that a
  value is returned.
- Document each element when returning a tuple.

For exceptions:

- Add `Raises:` only for deliberate, caller-relevant exceptions raised by the
  callable.
- Begin each entry with the concrete exception type.
- State the condition that causes the exception.
- Do not use a generic `Exception` entry when a more precise exception is
  raised.
- Do not document arbitrary exceptions that may propagate from dependencies.

### Classes and methods

Prefer docstrings on methods and functions. Do not add a class docstring merely
to repeat the constructor arguments or the class name. Put construction details
on `__init__` and operational details on the methods that implement them. Add a
class docstring only when there is class-level behavior, lifecycle information,
or an invariant that cannot be explained clearly by the individual methods.

### Complete example

```python
def load_sweep(path: Path, strict: bool = True) -> tuple[np.ndarray, float]:
    """
    Load sweep values and their point spacing from disk.

    Args:
        path (Path): Location of the saved sweep file.
        strict (bool): Whether malformed sweep metadata should cause an error.

    Returns:
        tuple[np.ndarray, float]: Sweep values followed by their point spacing
            in seconds.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If strict validation is enabled and the metadata is
            malformed.

    """
```

Indent continuation lines by four additional spaces so they remain attached to
the corresponding argument, return value, or exception in generated
documentation.

### Tests

Tests are exempt from these docstring requirements. Test names should normally
express the behavior being verified. A module or test docstring may still be
used when the setup or reasoning needs additional explanation. When a test uses
a docstring, retain the triple-quote layout described above.

## Sphinx integration

QCUtils uses Google-style docstrings so Sphinx renders the sections through
Napoleon. This is already configured in `docs/source/conf.py`, which enables
`sphinx.ext.autodoc` and `sphinx.ext.napoleon` with `napoleon_google_docstring =
True` (and NumPy-style docstrings switched off).

The API reference is generated from these docstrings: each page under
`docs/source/api/` is a short `automodule` stub, so documenting a new public
callable means writing its docstring, not editing the docs. A new *module* does
need a stub and a `toctree` entry.

Malformed docstrings surface as Sphinx warnings rather than silent bad output,
so build the docs after a substantial docstring change:

```bash
uv sync --group docs
uv run sphinx-build -W --keep-going -M html docs/source docs/build
```

Watch for reStructuredText traps inside docstrings. Indented text that is meant
to be literal needs a `::` marker, or docutils parses it as a block quote and
warns about the `*` in a signature; nested bullet lists need a blank line before
the sub-list.
