# Getting started

Install Skykit and the dependencies used by its HEALPix and plotting modules:

```console
pip install . healpy astropy matplotlib
```

The following example extracts and apodizes tiles from a small HEALPix map, builds a Morlet filter bank, and computes scattering coefficients. Replace the generated map with a map read from a FITS file for real data.

```python
import healpy as hp
import numpy as np

from skykit import Scattering2D, generate_filter_bank, healpix2tiles
from skykit.tile_utils import apply_apodization

nside = 32
sky_map = np.random.default_rng(0).normal(size=hp.nside2npix(nside))
tiles = healpix2tiles(sky_map, nside=nside, tile_nside=16, margin=4)
tiles = apply_apodization(tiles, taper_type="nuttall")

size = tiles.tile_full
filters = generate_filter_bank(size, size, J=2, L=4)
scattering = Scattering2D(filters, max_order=2)

# Full-resolution, lowpass-averaged coefficient maps.
maps = scattering.transform_tile(tiles.get_tile(0, 0, 0))

# One spatial mean per scattering path for every tile.
tile_values = scattering.transform_tileset(tiles, spatial_average=True)

# Also return unaveraged modulus feature maps U1 and U2 when needed.
with_features = scattering.transform_tile(
    tiles.get_tile(0, 0, 0), return_feature_maps=True
)
```

`S0`, `S1`, and `S2` are lowpass-averaged scattering outputs. The optional `U1` and `U2` arrays are feature maps before lowpass averaging. The spatial mean covers the full tile, including its margins.
