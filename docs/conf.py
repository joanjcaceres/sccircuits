"""Sphinx configuration for the SCCircuits documentation."""

from __future__ import annotations

import os
import sys
from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parent
ROOT = DOCS_DIR.parent
BUILD_DIR = DOCS_DIR / "_build"
MPLCONFIGDIR = BUILD_DIR / ".mplconfig"

BUILD_DIR.mkdir(exist_ok=True)
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.insert(0, str(ROOT))

project = "SCCircuits"
author = "Joan Caceres"
copyright = "2026, Joan Caceres"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
root_doc = "index"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_preserve_defaults = True

napoleon_google_docstring = False
napoleon_numpy_docstring = True

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
]
myst_heading_anchors = 3

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
