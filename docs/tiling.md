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

## Example: a tile near the north pole

The following figures come from the 353 GHz dust intensity map in `examples/healpix2tiles.ipynb`. The notebook extracts the tile containing Galactic longitude 280° and latitude 90° from a RING-ordered `nside=2048` map. Its interior is 256 × 256 pixels, with a 64-pixel margin on each side, so each plotted array is 384 × 384 pixels. The thin square marks the interior; everything outside it is margin. The axes are tile pixel coordinates, not Galactic longitude and latitude.

With the map loaded as `map_hpx`, the notebook makes both versions of that tile like this:

```python
import cmocean as cmo
import skykit as sk

geometric = sk.healpix2tiles(
    map_hpx, 2048, 256, 64, margin_method="geometric"
)
topological = sk.healpix2tiles(
    map_hpx, 2048, 256, 64, margin_method="topological"
)
face, tx, ty = geometric.tile_containing(280.0, 90.0)

fig = sk.plot_tile_flat(
    geometric, face, tx, ty, dpi=200,
    title="Tile interior and margin (geometric)",
    cmap=cmo.cm.balance, vmin=0, vmax=1.5e-3,
)
fig = sk.plot_tile_flat(
    topological, face, tx, ty, dpi=200,
    title="Tile interior and margin (topological)",
    cmap=cmo.cm.balance, vmin=0, vmax=1.5e-3,
)
```

The important issue to understand is that for `topological` margin method, we use the pixels of the neighboring HEALPix tile. In some regions, like the poles of the map it will cause stretching of features as orientation of axes changes from between the faces. The `geometric` method tries to overcome this by extrapolating the coordinates from the edges of the interior region. Then it interpolates the correct map value for the coordinate. This has small issues at the corners where four faces meet. For both methods, we note that the region inside the margins are one-to-one mapping of the HEALPix superpixel. Hence there are no impacts from projection, interpolation etc. 

```{figure} figures/dust_north-pole-tile_geometric.png
:alt: North pole dust intensity tile with geometric margins and its interior outlined.
:width: 85%

**Geometric margins.** The pixels outside the outlined interior follow the local geometric continuation of the tile.
```

```{figure} figures/dust_north-pole-tile_topological.png
:alt: The same north pole dust intensity tile with topological margins and its interior outlined.
:width: 85%

**Topological margins.** The interior is the same, while the margin samples are chosen by walking across HEALPix face boundaries. Both plots use the same color range, so differences outside the square can be compared directly.
```

For a Q/U map, set `pol=True`. The notebook averages the even- and odd-ring 353 GHz maps, smooths them by 7.5 arcminutes, and extracts a topological tile with a 128-pixel margin. The resulting array is 512 × 512 pixels per Stokes component. Given the prepared Q/U array as `qu_map`, the corresponding calls are:

```python
qu_tiles = sk.healpix2tiles(qu_map, 2048, 256, 128, pol=True)
face, tx, ty = qu_tiles.tile_containing(280.0, 90.0)
fig = sk.plot_tile_flat(
    qu_tiles, face, tx, ty, dpi=200,
    title="North pole polarization tile (topological)",
    cmap=cmo.cm.balance,
)
```

```{figure} figures/pol-dust_north-pole-tile_topological.png
:alt: Topological north pole polarization tile, with Stokes Q and U shown side by side and their interiors outlined.
:width: 100%

**Polarisation tile.** `plot_tile_flat` displays Stokes Q and U side by side, with separate color scales. The outlined 256 × 256 interiors are surrounded by the 128-pixel margins.
```

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
