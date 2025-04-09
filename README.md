# QCUtils

QCUtils simplifies measurement workflows and replaces QCoDeS for handling measurements.

## Preparation

We manage and run our measurements using the tool `uv`. To get started, follow the official [Astral installation guide](https://astral.sh/uv/) to install `uv`. It handles virtual environments and package dependencies for you, simplifying Python package management.

### Initializing a Measurement

To start a measurement, first initialize a `uv` project with the desired measurement name:

```sh
uv init measurement_name
cd measurement_name
```

This command automatically creates a virtual environment and installs the necessary dependencies. To add additional packages to your project, use the `uv add` command. For example, to add commonly used packages, like `qcutils`, you can run:

```sh
uv add git+https://gitlab.com/squad-lab/qcutils        # Install qcutils
uv add git+https://gitlab.com/squad-lab/measurements/drivers # Install squad drivers
uv add qcodes_contrib_drivers                         # Install additional drivers
```

> [!note] NOTE  
> The `qcutils`, `qcodes`, `zhinst`, and `zhinst-qcodes` packages are all managed by `qcutils`, so there's no need to install them separately.

### Running Commands

Instead of manually activating a virtual environment and running `python -m package.method`, you can use `uv` to streamline this:

```sh
uv run -m package.method
```

### Using pip with `uv`

Although `uv` supports native `pip` functionality, we recommend using the `uv add` method for consistency:

```sh
uv pip install package
```

### Using `pyproject.toml`

If you prefer to skip the `add` commands, you can set up the project directly with a `pyproject.toml` file. Here’s an example:

```toml
name = "example"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "drivers",
    "qcodes-contrib-drivers>=0.23.0",
    "qcutils",
]

[tool.uv.sources]
qcutils = { git = "https://gitlab.com/squad-lab/qcutils" }
drivers = { git = "https://gitlab.com/squad-lab/measurements/drivers" }
```

After updating the `pyproject.toml` file, you can sync the dependencies with:

```sh
uv sync
```

This will automatically fetch and install the required packages.

## Measurement

For an example of using `qcutils` for a measurement, refer to [example.py](example.py). API documentation is currently available in the code itself.

## Development

We welcome contributions to `qcutils`! To get started with development, clone the repository:
```sh
git clone https://gitlab.com/squad-lab/qcutils
```
Navigate into the cloned directory and install the package in development mode:
```sh
cd qcutils
uv add --dev .
```

## Authors

- Spandan Anupam: [s.anupam@fz-juelich.de](mailto:s.anupam@fz-juelich.de)
- Lino Visser: [l.visser@fz-juelich.de](mailto:l.visser@fz-juelich.de)