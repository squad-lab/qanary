# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
#
# Adapted from the Qimchi documentation configuration.

import importlib.metadata

# ----- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Qanary"
copyright = "2024-%Y, Spandan Anupam, Jyotirmaya Shivottam"
author = "Spandan Anupam, Jyotirmaya Shivottam"

# Read the version from installed package metadata instead of pinning a
# literal, so a bump in pyproject.toml never has to be repeated here.
#
# Import the module rather than `from importlib.metadata import version`: every
# name left in this file's namespace is read as a Sphinx config value, so that
# form would bind `version` to a *function* and the build dies writing
# objects.inv with "expected string or bytes-like object, got 'function'".
try:
    release = importlib.metadata.version("qanary")
except importlib.metadata.PackageNotFoundError:  # built without the package
    release = "0.0.0"
version = release

# ----- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

needs_sphinx = "8.2.0"  # Minimum Sphinx version required
extensions = [
    "sphinx.ext.autodoc",  # Automatically generate documentation from docstrings
    "sphinx.ext.napoleon",  # Support for Google-style and NumPy-style docstrings
    "sphinx.ext.todo",  # Support for TODOs in the documentation
    "sphinx.ext.mathjax",  # Support for LaTeX math in the documentation
    "sphinx.ext.intersphinx",  # Link to other projects' documentation
    "sphinx.ext.autosummary",  # Automatically generate summary tables for modules and classes
    "sphinx.ext.viewcode",  # Link to source code in the documentation
    "myst_parser",  # Support for Markdown files in the documentation
    "sphinx_copybutton",  # Copy button on code blocks
    "sphinx_design",  # Cards, tabs and grids in MyST
]

templates_path = ["_templates"]  # Path to templates used for HTML output
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]  # Patterns to exclude from the build
html_static_path = ["_static"]  # Path to static files used for HTML output

# ----- Options for Theme customization -----------------------------------------

html_theme = "furo"
html_title = "Qanary"
# The project avatar, with its white background flood-filled to transparent so
# it does not sit in a white box against Furo's dark sidebar, and trimmed to its
# content so it fills the sidebar slot.
html_logo = "_static/assets/qanary-logo.png"
# Kept square and untrimmed -- a letterboxed tab icon renders squashed.
html_favicon = "_static/assets/qanary-icon.png"
html_copy_source = True
html_last_updated_fmt = ""

html_css_files = [
    "custom.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/fontawesome.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/solid.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/brands.min.css",
]
html_theme_options = {
    "footer_icons": [
        {
            "name": "GitLab",
            "url": "https://gitlab.com/squad-lab/qanary",
            "html": "",
            "class": "fa-brands fa-gitlab",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/qanary/",
            "html": "",
            "class": "fa-brands fa-python",
        },
    ],
}

# ----- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

add_function_parentheses = (
    True  # Add parentheses to function names (e.g. `function()` instead of `function`)
)
add_module_names = True  # Add module names to class and function names (e.g. `module.ClassName` instead of `ClassName`)
highlight_language = "python"  # default language to highlight code blocks
pygments_style = "sphinx"  # pygments style to use for syntax highlighting
autoclass_content = "both"  # Include both class and __init__ docstring
autodoc_member_order = "bysource"  # Order members by source order
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}  # Suffixes of source files
master_doc = "index"  # The master toctree document.
htmlhelp_basename = "qanarydoc"  # Name of the help file (without suffix)
html_show_sourcelink = True  # Show source link in HTML output
html_show_copyright = True  # Show copyright in HTML output
html_show_search_summary = True  # Show search summary in HTML output
autodoc_typehints = "description"

# NOTE: Qanary imports instrument drivers that are not installed on a docs
# runner. Without these, autodoc fails to import the modules it documents.
autodoc_mock_imports = [
    "drivers",
    "zhinst",
    "zhinst_qcodes",
]

# The codebase uses Google-style docstrings exclusively.
napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {  # Links to other projects' documentation
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "zarr": ("https://zarr.readthedocs.io/en/stable/", None),
    "qcodes": ("https://microsoft.github.io/Qcodes/", None),
}

# ----- MyST --------------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "linkify",
    "substitution",
]
myst_heading_anchors = 3
