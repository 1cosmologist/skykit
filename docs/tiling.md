# HEALPix tiling

Skykit divides each of the 12 HEALPix faces into square, nonoverlapping interiors and adds a margin around each interior. The result is a set of overlapping tiles that can be filtered with two-dimensional FFTs. Tile coordinates are face pixel indices, so equal steps on a tile do not always represent equal angular distances on the sky.

## Choose the geometry

- `nside` is the HEALPix map resolution and must be a power of two.
- `tile_nside` is the side length of each tile **interior** in pixels and must divide `nside`. There are `12 * (nside // tile_nside) ** 2` tiles.
- `margin` adds that many pixels on every side. A full tile is `(tile_nside + 2 * margin) ** 2` pixels. The implementation requires `margin < nside`; use a nonnegative margin.
- `nested=False` expects the default HEALPix RING ordering. Pass `nested=True` for a NESTED input map.
- `pol=True` expects a `(2, 12 * nside**2)` array of Stokes Q and U in the **HEALPix/COSMO** convention. For the same reference axes, `U_COSMO = -U_IAU`. Skykit does not convert conventions.

The tile index order is face, then face-local `tx`, then `ty`. Scalar tile data have shape `(n_tiles, tile_full, tile_full)`; polarisation adds a size-two Q/U axis after the tile axis.

## Choose a margin method

Both methods assign the same exact HEALPix pixels to tile interiors. They differ only in how they fill margins beyond a face boundary.

| `margin_method` | How the margin is filled | When to choose it | Limitation |
| --- | --- | --- | --- |
| `"topological"` (default) | Walks each out-of-face tile coordinate across HEALPix face boundaries in x-then-y and y-then-x order. If the two walks select different pixels near a corner, it averages their map values. | Use when you want margins based on face adjacency and direct map samples. | At ambiguous corners, the average is a synthetic value rather than the value of one HEALPix pixel. A flat square grid cannot be globally seamless at the face vertices. |
| `"geometric"` | Extrapolates interior pixel-centre vectors in 3D, renormalizes them onto the sphere, then assigns the nearest HEALPix pixel with `vec2pix`. | Use when you want margins to follow a local geometric continuation of the tile. | Extrapolation and nearest-pixel assignment are approximate; a margin may repeat or skip source pixels. |

For polarisation, each sampled Q/U pair is rotated into a frame obtained by parallel-transporting the tile centre's north direction to that pixel. This is done for either margin method. The returned Q/U remain in the COSMO convention.

## Extract, process, and reconstruct

```python
import healpy as hp
import numpy as np

from skykit import HealpixTileProjector
from skykit.tile_utils import apply_apodization

nside = 32
sky_map = np.random.default_rng(0).normal(size=hp.nside2npix(nside))

# Cache the pixel mapping if several maps share the same geometry.
projector = HealpixTileProjector(
    nside, tile_nside=16, margin=4, margin_method="topological"
)
tiles = projector.map2tiles(sky_map, nested=False)
processed = apply_apodization(tiles, taper_type="cosine")

# Reconstruction uses interiors only; overlapping margins are discarded.
reconstructed = projector.tiles2map(tiles, nested=False)
```

For a single map, {py:func}`skykit.healpix_tiling.healpix2tiles` creates the projector for you. {py:func}`skykit.healpix_tiling.tiles2healpix` reconstructs a map from a `TileSet`. Reconstruction is exact for an unmodified scalar tile set because each HEALPix pixel appears once in the interiors. Apodization keeps interior weights at one and tapers only the margins.

FFT convolution treats each tile as periodic. Choose a margin wide enough for the wavelet and lowpass support, then apodize the margin to reduce edge artifacts. See the {doc}`wavelets-scattering` guide for the filter scales and output conventions.
