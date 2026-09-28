# Skykit

Skykit turns HEALPix maps into overlapping square tiles and computes two-dimensional wavelet scattering transforms on them. It supports scalar maps and COSMO-convention Stokes Q/U maps.

**Documentation:** [skykit.readthedocs.io](https://skykit.readthedocs.io/)

## What it provides

- HEALPix tiling with topological or geometric margin assignment, and reconstruction from tile interiors.
- Fourier-domain Morlet, Gabor, and bump filter banks.
- JAX first- and second-order wavelet feature maps with scale and orientation paths.
- Mean, variance, and cross scattering statistics from feature maps.
- HDF5 feature-map storage, apodization windows, HDF5 tile storage, and plotting helpers.

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

from skykit import (apply_apodization, generate_filter_bank, healpix2tiles,
                    scattering_transform, wavelet_transform_tile)

nside = 32
sky_map = np.random.default_rng(0).normal(size=hp.nside2npix(nside))
tiles = healpix2tiles(sky_map, nside=nside, tile_nside=16, margin=4)
tiles = apply_apodization(tiles)

filters = generate_filter_bank(tiles.tile_full, tiles.tile_full, J=2, L=4)
single = wavelet_transform_tile(tiles.get_tile(0, 0, 0), filters, order=2)
single_stats = scattering_transform(single)

with wavelet_transform_tile(tiles, filters, order=2,
                            filepath="feature_maps.h5", batch_size=4) as features:
    statistics = scattering_transform(features)
    cross = scattering_transform(features, features)
```

`single` is a `FeatureMap`; `features` is a `FeatureMapSet`. Their `paths` map each `U1` or `U2` feature axis to its wavelet scales and orientations. `statistics` is a `ScatteringStatistics` object with corresponding `values` and `paths`. Use `reduction="variance"` for pixel variance and `operation1`/`operation2` for JAX functions applied before reduction. With two sets, `cross` contains only path-pair statistics. HDF5 files store the maps and paths together and are read in tile and path batches.

`healpix2tiles` expects RING ordering by default; use `nested=True` for a NESTED map. With `pol=True`, Q/U input must use the HEALPix/COSMO convention.

## Guides and API

- [HEALPix tiling and margin methods](https://skykit.readthedocs.io/en/latest/tiling.html)
- [Wavelet and scattering mathematics](https://skykit.readthedocs.io/en/latest/wavelets-scattering.html)
- [API reference](https://skykit.readthedocs.io/en/latest/api.html)

The scattering guide describes the wavelet paths and statistics workflow.

## Development

```bash
python -m pytest -q
python -m pip install -r docs/requirements.txt
sphinx-build -b html -W docs docs/_build/html
```
