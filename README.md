# Skykit

`skykit` is a Python package designed for high-performance sky tiling and 2D Wavelet Scattering Transforms on astrophysical maps, particularly those using HEALPix pixelation. Powered by `JAX`, it provides GPU-accelerated implementations for filtering and feature extraction without spatial downsampling (un-decimated), ensuring a 1:1 pixel correspondence with original patches.

## Table of Contents
- [Implemented Wavelets](#implemented-wavelets)
- [Scattering Transforms](#scattering-transforms)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Testing](#testing)

## Implemented Wavelets
The package generates 2D wavelets in Fourier space, allowing for precise frequency-band and orientation selection:
- **Morlet Wavelets:** The standard for most scattering pipelines. They remove the DC-bias from a plane-wave, separating signals at multiple rotational bands accurately. Excellent for general-purpose filament and orientation discovery on diffuse fields (e.g., ISM dust emission).
- **Gabor Wavelets:** The standard complex Gaussian-modulated sinusoid, providing optimal joint time-frequency localization.
- **Bump Wavelets:** Smooth, compactly supported wavelets in Fourier space. Ideal when strict avoidance of cross-talk between frequency bands is required.
- **Lowpass $\phi$:** Used to extract the translation-invariant average (or background) structure of the map.

## Scattering Transforms
The Wavelet Scattering Transform provides a stable, multi-scale, and rotation-covarient representation of images. By building a hierarchical cascade of wavelet convolutions and non-linear complex modulus operations, it can extract non-Gaussian features effectively:
- **$S_0$ (0th Order):** The local average of the signal (lowpass invariant energy).
- **$S_1$ (1st Order):** Edge and texture responses from wavelets at various scales and orientations.
- **$S_2$ (2nd Order):** Responses bridging interactions between mixed scales, capturing complex textures, filaments, and phase-coupling.

In `skykit`, `Scattering2D` handles these transforms with pre-compiled JAX operations over tiled subsets for optimal speed.

## Installation
The package is built using the `meson-python` build backend. Ensure you have a recent version of `pip` and build tools installed.

```bash
git clone https://github.com/1cosmologist/skykit.git
cd skykit
pip install .
```
Dependencies: `numpy`, `jax`, `jaxlib`. (You may also want `healpy` and `matplotlib` if exploring the notebooks).

## Usage
Extract HEALPix maps to overlapping squared tiles, apply an apodization window, and compute scattering coefficients:

```python
import healpy as hp
from skykit import healpix2tiles, generate_filter_bank, Scattering2D
from skykit.tile_utils import apply_apodization

# Load HEALPix map
nside = 2048
map_hpx = hp.read_map('my_map.fits')

# Create Overlapping Patches (e.g., 256x256 with a 64-pixel margin)
tiles = healpix2tiles(map_hpx, nside=nside, tile_nside=256, margin=64)

# Apodize (Nuttall window) to tie boundaries to zero
apodized_tiles = apply_apodization(tiles, taper_type='nuttall', taper_width=64)
patch = apodized_tiles.get_tile(face=0, tx=0, ty=0)

# Apply 2D Scattering Transform
# J = max scales, L = orientations, max_order = depth of transform
filters = generate_filter_bank(M=384, N=384, J=3, L=8, wavelet_type='morlet')
scatter = Scattering2D(filters, max_order=2)
coeffs = scatter.transform_tile(patch)

# Access Coefficients
S0 = coeffs['S0']
S1 = coeffs['S1']
S2 = coeffs['S2']

# One spatial mean per scattering path, over the full tile
tile_coeffs = scatter.transform_tile(patch, spatial_average=True)

# Include unaveraged modulus feature maps alongside the scattering coefficients
with_features = scatter.transform_tile(patch, return_feature_maps=True)
U1 = with_features['U1']
U2 = with_features['U2']
```

## Examples
A detailed walkthrough is provided in the Jupyter Notebook: `examples/test_scattering.ipynb`.
It covers filter-bank visualization, tile extraction on real maps (Planck PR3), transform computation, and coefficient visualization ($S_0$, $S_1$, $S_2$).

## Testing
`skykit` uses Python's standard `unittest` framework to verify module logic. To run the tests, execute:
```bash
python -m unittest discover -s tests
```
