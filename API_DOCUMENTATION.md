# skykit API Documentation & Mathematical Foundations

Welcome to the `skykit` documentation. This file provides a comprehensive API reference for all public classes and functions, paired closely with the definitions of the core mathematical concepts driving the library.

---

## API Reference & Mathematical Definitions

### `skykit.healpix_tiling`

Analyzing spherical maps (like CMB data) using 2D convolutional networks or scattering transforms requires projecting the curved sphere onto flat domains. `skykit` decomposes HEALPix maps into an overlapping grid of square patches centered on HEALPix face sub-divisions.

**Topological Unwrapping:**
To properly handle boundary margins without interpolation, `skykit` uses exact topological boundary unfolding (written in JAX). Out-of-bounds coordinates on a tile are iteratively wrapped across HEALPix boundaries to their exact corresponding pixels on adjacent faces.

**Spin-2 Polarisation (Q/U):**
Polarisation is a spin-2 field defined relative to the local meridian (North). When projecting onto a flat tile, `skykit` parallel-transports the Q and U Stokes parameters from each pixel's local frame to a unified reference frame at the tile's center. A pixel's rotation angle $\psi$ is computed via Rodriguez's rotation of the center's normal vector. 
For a counter-clockwise frame rotation $\psi$, the Stokes parameters transform as:
$$ Q' = Q \cos(2\psi) + U \sin(2\psi) $$
$$ U' = -Q \sin(2\psi) + U \cos(2\psi) $$

#### `class HealpixTileProjector`
Pre-computes and caches pixel mappings and polarization rotation angles for HEALPix map transformations. Prevents recalculating arrays when transforming multiple maps with the same geometry.
- `__init__(nside: int, tile_nside: int, margin: int, pol: bool = False)`
  - `nside`: HEALPix resolution parameter (power of 2) of the target map.
  - `tile_nside`: Interior side length of each tile. Must divide `nside`.
  - `margin`: Overlap border width appended around the interior. Must be `< nside`.
  - `pol`: If `True`, computations accommodate an additional polarisation dimension and calculate parallel-transport $\psi$ matrices.
- `map2tiles(healpix_map: np.ndarray, nested: bool = False) -> TileSet`:
  - `healpix_map`: The 1D input array (scalar) or 2D shape `(2, npix)` input trace for `[Q, U]`.
  - `nested`: Selects `nest=True` or `ring` map ordering logic.
  - *Returns*: A `TileSet` containing canonical projections.
- `tiles2map(tileset: TileSet, nested: bool = False) -> np.ndarray`:
  - `tileset`: A `TileSet` previously produced by `map2tiles`. Only interior pixels are used; border margins are discarded.
  - `nested`: If `True`, returns the output array in NESTED pixel ordering rather than the default RING ordering.
  - *Returns*: A reconstructed 1D array of length `npix` (scalar) or shape `(2, npix)` (polarisation).

#### `healpix2tiles(healpix_map: np.ndarray, nside: int, tile_nside: int, margin: int, pol: bool = False, nested: bool = False)`
Standalone wrapper that initializes a `HealpixTileProjector` and dynamically decomposes an incoming map. Best used on one-off requests.

#### `tiles2healpix(tileset: TileSet, nested: bool = False)`
Standalone wrapper that unpacks a `TileSet` struct and reconstructs a HEALPix map.

---

### `skykit.tileset`

#### `class TileSet`
A container for the overlapping square tiles extracted from a HEALPix map.
- `__init__(data: np.ndarray, nside: int, tile_nside: int, margin: int, pol: bool = False)`
  - `data`: 3D scalar array of shape `(N_tiles, full_w, full_w)` or 4D polarisation array of shape `(N_tiles, 2, full_w, full_w)`, where `full_w = tile_nside + 2 * margin`.
- **Properties**: `data` (4D/3D array), `nside`, `tile_nside`, `margin`, `pol`
- **Methods**:
  - `tile_index(face, tx, ty) -> int`: Returns the linear scalar index `((face * n_subtiles) + tx) * n_subtiles + ty`.
  - `get_tile(face, tx, ty) -> ndarray`: Returns full tile data array chunk (interior + boundaries) given index parameters.
  - `get_interior(face, tx, ty) -> ndarray`: Returns strictly the inner un-padded array segment.
  - `tile_center(face, tx, ty) -> tuple(float, float)`: Returns spatial center geometry `(lon, lat)` in degrees.
  - `tile_containing(lon, lat) -> tuple(int, int, int)`: Validates coordinate projection constraints and returns `(face, tx, ty)` bounding the `(lon, lat)`.
  - `iter_tiles() -> Generator`: Linearly yields tuples `(face, tx, ty, tile_data)`.

---

### `skykit.tile_utils`

#### `create_apodization_window(tile_nside: int, margin: int, taper_width: int = None, taper_type: str = 'cosine')`
Creates a 2D tapering window to strongly suppress FFT boundary artifacts for Scattering logic.
- `tile_nside`: Interior spatial bounding size (window guarantees 1.0 logic here).
- `margin`: Defines padding bounds (tapers drop uniformly to 0.0).
- `taper_width`: Overrides strict margin taper logic limit (must be $\leq$ `margin`).
- `taper_type`: Determines the falloff formula. Choices include `'nuttall'`, `'cosine'`, `'sine'`.

#### `apply_apodization(tileset: TileSet, taper_width: int = None, taper_type: str = 'cosine', inplace: bool = False)`
Applies the generated window over every tile in a `TileSet`.
- `tileset`: The input `TileSet` whose data arrays will be tapered.
- `taper_width`: Width of the taper transition in pixels. Defaults to `tileset.margin`. Must be `<= margin`.
- `taper_type`: Taper function used in the window. Choices include `'nuttall'`, `'cosine'`, `'sine'`.
- `inplace`: If `True`, modifies `tileset.data` directly to avoid a copy. If `False` (default), returns a new `TileSet`.

#### `write_tileset_hdf5(tileset: TileSet, filepath: str)`
Serializes a `TileSet` to an HDF5 file, storing the `data` array and geometry metadata (`nside`, `tile_nside`, `margin`, `pol`) as HDF5 attributes.
- `tileset`: The `TileSet` instance to serialize.
- `filepath`: Destination path for the output `.h5` file.

#### `read_tileset_hdf5(filepath: str)`
Given a strict path parameter context, reads an HDF5 file, validates its attributes formally (`nside`, `tile_nside`, `margin`, `pol`), and returns an accurate `TileSet`.

---

### `skykit.jax_wavelets`

The scattering transform relies on bandpass wavelets $\psi_{j, \theta}$ and lowpass filters $\phi$. Because we perform convolutions via FFTs, these filters are evaluated intrinsically on a normalized 2D frequency grid $(u, v) \in [-0.5, 0.5]$ of size `(M, N)`.

First, the coordinates are rotated by angle $\theta$:
$$ u_{\text{rot}} = u \cos\theta + v \sin\theta $$
$$ v_{\text{rot}} = -u \sin\theta + v \cos\theta $$

The base parameters defining the filter shape in $k$-space are:
- `sigma` ($\sigma$): Bandwidth of the filter.
- `xi` ($\xi$): Central target frequency.
- `slant` ($s$): Aspect ratio (ellipticity) of the filter envelope.

#### `generate_filter_bank(M: int, N: int, J: int, L: int, wavelet_type: str = 'morlet', sigma0: float = 0.8, xi0: float = np.pi/4.0, slant: float = 0.5)`
Constructs a complete dictionary of frequency-space wavelets (`psi`) and a lowpass filter (`phi`).
- `M`, `N`: Dimensions of the expected input Fourier images.
- `J`: Number of dyadic scale levels. Scale $j$ ($0 \leq j < J$) has bandwidth $\sigma_j = \sigma_0 \cdot 2^j$ and peak frequency $\xi_j = \xi_0 / 2^j$.
- `L`: Number of discrete orientations uniformly covering $[0, \pi)$. Orientation $l$ has angle $\theta_l = l\pi / L$.
- `wavelet_type`: Choice of wavelet architecture. One of `'morlet'`, `'gabor'`, or `'bump'`.
- `sigma0`: Base spatial bandwidth parameter. Dilated geometrically by $2^j$ at each scale.
- `xi0`: Base peak frequency (in normalised units, i.e. fractions of $\pi$). Halved at each successive scale.
- `slant`: Ellipticity (aspect ratio) of the filter envelope. Values $< 1$ elongate the filter along the orientation axis.
- *Returns*: A dict with keys `'psi'` (list of dicts, each with `'j'`, `'theta'`, `'val'`) and `'phi'` (dict with `'j'`, `'val'`).

#### Individual Filter Generators:
Low-level JAX functions that return a 2D real-valued array on the `(M, N)` FFT frequency grid. All share the arguments `M`, `N`, `sigma`, `theta`, `xi`, and `slant` (where applicable), matching the symbols defined above.

- **`gabor_2d(M, N, sigma, theta, xi, slant=0.5)`**: A Gaussian envelope centred at frequency $\xi$ along orientation $\theta$. Strictly non-negative in $k$-space:
  $$ \hat{\psi}_{\text{Gabor}}(u, v) = \exp\left( - \frac{(u_{\text{rot}} - \xi)^2 + (v_{\text{rot}} / s)^2}{2 \sigma^2} \right) $$
- **`morlet_2d(M, N, sigma, theta, xi, slant=0.5)`**: A Gabor wavelet corrected for exact zero mean. Subtracts a DC-centred Gaussian scaled by $K$ to enforce the admissibility condition $\hat{\psi}(0,0) = 0$, preventing low-frequency background leakage:
  $$ \hat{\psi}_{\text{Morlet}}(u, v) = \hat{\psi}_{\text{Gabor}}(u, v) - K \exp\left( - \frac{u_{\text{rot}}^2 + (v_{\text{rot}} / s)^2}{2 \sigma^2} \right) $$
  where $K = \exp\!\left(-\xi^2 / (2\sigma^2)\right)$ ensures $\hat{\psi}_{\text{Morlet}}(0,0) = K - K \cdot 1 = 0$. The subtracted term creates a small negative region near the DC origin.
- **`bump_2d(M, N, sigma, theta, xi, slant=0.5)`**: A compactly supported filter with exactly zero energy outside the unit ball. Defining the normalised squared distance $d^2 = [(u_{\text{rot}} - \xi)^2 + (v_{\text{rot}} / s)^2] / \sigma^2$:
  $$ \hat{\psi}_{\text{Bump}}(u, v) = \begin{cases} e \cdot \exp\!\left(\dfrac{1}{d^2 - 1}\right) & \text{if } d^2 < 1 \\ 0 & \text{otherwise} \end{cases} $$
  The factor of $e$ normalises the peak to 1.
- **`lowpass_2d(M, N, sigma)`**: An isotropic Gaussian $\hat{\phi}$ centred at DC, capturing the residual low-frequency energy not covered by any $\psi_{j,\theta}$:
  $$ \hat{\phi}(u, v) = \exp\left( - \frac{u^2 + v^2}{2 \sigma^2} \right) $$

---

### `skykit.scattering_transform`

The scattering transform extracts translation-invariant, deformation-stable features through a cascade of wavelet convolutions and pointwise modulus non-linearities:
- **Zeroth Order ($S_0$):** Low-pass averaged signal energy: $S_0 = x \ast \phi_J$
- **First Order ($S_1$):** Edge and filament response per $(j_1, \theta_1)$ path: $S_1[j_1, \theta_1] = |x \ast \psi_{j_1, \theta_1}| \ast \phi_J$
- **Second Order ($S_2$):** Cross-scale modulation between $(j_1, \theta_1)$ and $(j_2, \theta_2)$ with $j_2 > j_1$: $S_2[j_1, \theta_1, j_2, \theta_2] = \bigl||x \ast \psi_{j_1, \theta_1}| \ast \psi_{j_2, \theta_2}\bigr| \ast \phi_J$

All convolutions are performed using JAX-accelerated FFTs.

#### `class Scattering2D`
JAX-accelerated class to compute the 2D Wavelet Scattering Transform.
- `__init__(M: int, N: int, J: int, L: int, wavelet_type: str = 'morlet')`
  - `M`, `N`: Spatial dimensions of the input tiles the transform will be applied to.
  - `J`: Number of dyadic scales (must match the filter bank used for any pre-computed tiles).
  - `L`: Number of orientations per scale.
  - `wavelet_type`: Wavelet family used to build the internal filter bank via `generate_filter_bank()`.
  - On construction, pre-computes and stores the stacked Fourier-space filter arrays `psi_vals` (shape `(J*L, M, N)`) and `phi_val` (shape `(M, N)`).
- **Methods**:
  - `transform_tile(x: jnp.ndarray) -> dict`:
    - `x`: Input 2D real-valued spatial array of shape `(M, N)`.
    - *Returns*: A dictionary with keys:
      - `'S0'`: ndarray of shape `(M, N)` — zeroth-order lowpass averaged map.
      - `'S1'`: ndarray of shape `(J*L, M, N)` — first-order coefficients indexed by `(j, l)` in row-major order.
      - `'S2'`: ndarray of shape `(n_paths, M, N)` — second-order coefficients for all valid `(j1, l1, j2, l2)` pairs with `j2 > j1`.
