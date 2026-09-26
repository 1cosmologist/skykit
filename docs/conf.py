"""Sphinx configuration for the Skykit documentation."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "Skykit"
author = "Shamik Ghosh"
release = (ROOT / "VERSION").read_text(encoding="utf-8").strip()

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]
myst_enable_extensions = ["dollarmath"]
myst_heading_anchors = 3
autodoc_mock_imports = ["healpy", "jax", "jaxlib", "h5py", "astropy", "matplotlib"]
autodoc_default_options = {"member-order": "bysource"}

html_theme = "shibuya"
html_title = f"Skykit {release}"

html_theme_options = {
    "page_layout": "compact",
    "accent_color": "ruby",
    "nav_links": [
        {"title": "Home", "url": "index"},
        {"title": "HEALPix tiling", "url": "tiling"},
        {"title": "Wavelets & scattering", "url": "wavelets-scattering"},
        {
            "title": "API reference",
            "url": "api",
            "children": [
                {"title": "API overview", "url": "api"},
                {"title": "HEALPix tiling", "url": "api/healpix_tiling"},
                {"title": "TileSet", "url": "api/tileset"},
                {"title": "Tile utilities", "url": "api/tile_utils"},
                {"title": "Wavelet filters", "url": "api/jax_wavelets"},
                {"title": "Scattering transform", "url": "api/scattering_transform"},
                {"title": "Plotting", "url": "api/tile_plotting"},
            ],
        },
    ],
}

templates_path = ["_templates"]

exclude_patterns = ["_build"]
