# Skykit

Skykit turns HEALPix maps into overlapping square tiles and computes two-dimensional wavelet scattering transforms on them. It supports scalar maps and COSMO-convention Stokes Q/U maps.

**Documentation:** [skykit.readthedocs.io](https://skykit.readthedocs.io/)

## What it provides

- HEALPix tiling with topological or geometric margin assignment, and reconstruction from tile interiors.
- Fourier-domain Morlet, Gabor, and bump filter banks.
- JAX scattering coefficients at orders 0, 1, and 2, as full-resolution maps or one spatial mean per path and tile.
- Optional unaveraged modulus feature maps, apodization windows, HDF5 tile storage, and plotting helpers.

## Install

```bash
git clone https://github.com/1cosmologist/skykit.git
cd skykit
python -m pip install . healpy astropy matplotlib
```

The package imports its HEALPix and plotting modules at startup, so `healpy`, `astropy`, and `matplotlib` are needed alongside the dependencies declared in `pyproject.toml`.

## Quick start

```python
import healpy as hp
import numpy as np

from skykit import Scattering2D, apply_apodization, generate_filter_bank, healpix2tiles

nside = 32
sky_map = np.random.default_rng(0).normal(size=hp.nside2npix(nside))
tiles = healpix2tiles(sky_map, nside=nside, tile_nside=16, margin=4)
tiles = apply_apodization(tiles)

filters = generate_filter_bank(tiles.tile_full, tiles.tile_full, J=2, L=4)
scattering = Scattering2D(filters, max_order=2)

coefficient_maps = scattering.transform_tile(tiles.get_tile(0, 0, 0))
tile_values = scattering.transform_tileset(tiles, spatial_average=True)
```

`coefficient_maps` contains lowpass-averaged `S0`, `S1`, and `S2` maps. `tile_values` contains one mean over the full tile for each scattering path. Pass `return_feature_maps=True` to either transform method to also obtain the unaveraged `U1` and `U2` maps.

`healpix2tiles` expects RING ordering by default; use `nested=True` for a NESTED map. With `pol=True`, Q/U input must use the HEALPix/COSMO convention.

## Guides and API

- [HEALPix tiling and margin methods](https://skykit.readthedocs.io/en/latest/tiling.html)
- [Wavelet and scattering mathematics](https://skykit.readthedocs.io/en/latest/wavelets-scattering.html)
- [API reference](https://skykit.readthedocs.io/en/latest/api.html)

The [example notebook](examples/test_scattering.ipynb) shows the workflow on astrophysical maps.

## Development

```bash
python -m pytest -q
python -m pip install -r docs/requirements.txt
sphinx-build -b html -W docs docs/_build/html
```
