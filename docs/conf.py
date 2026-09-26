"""Sphinx configuration for the Skykit documentation."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

project = "Skykit"
author = "Shamik Ghosh"
release = (ROOT / "VERSION").read_text(encoding="utf-8").strip()

extensions = ["myst_parser", "sphinx.ext.mathjax"]
myst_enable_extensions = ["dollarmath"]
myst_heading_anchors = 3

html_theme = "sphinxawesome_theme"
html_title = f"Skykit {release}"

exclude_patterns = ["_build"]
