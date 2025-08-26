# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "temporal"
copyright = "2024, Your Name"
author = "Your Name"

try:
    from temporal import __version__
    version = __version__
    release = version
except ImportError:
    version = "0.0.0"
    release = "0.0.0"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "autoapi.extension",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# MyST configuration
myst_enable_extensions = [
    "deflist",
    "fieldlist",
    "attrs_block",
    "attrs_inline",
]

# AutoAPI configuration
ROOT = Path(__file__).resolve().parents[1]
autoapi_type = "python"
autoapi_dirs = [str(ROOT / "temporal")]
autoapi_root = "api"
autoapi_keep_files = True
autoapi_options = [
    "show-inheritance-diagram",
    "show-module-summary",
]


# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

try:
    import furo
    html_theme = "furo"
except ImportError:
    html_theme = "pydata_sphinx_theme"


html_static_path = ["_static"]
