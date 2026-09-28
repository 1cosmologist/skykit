# Wavelet filters and scattering transforms

Skykit constructs filters on each tile's **unshifted** two-dimensional FFT grid. The variables `u` and `v` are frequencies in cycles per pixel along the first and second tile-array axes. They are not spherical-harmonic multipoles or fixed angular frequencies on the sky.

## Fourier orientation convention

For orientation $\theta$, the code evaluates filters in the rotated frequency coordinates

$$
u_{\rm rot}=u\cos\theta+v\sin\theta,\qquad
v_{\rm rot}=-u\sin\theta+v\cos\theta.
$$

This is a change of coordinates, not a rotation of the input map. `u_rot` measures frequency along the wavelet's wavevector and `v_rot` measures frequency perpendicular to it. A filter centred at $(u_{\rm rot},v_{\rm rot})=(\xi,0)$ responds to oscillations along $\theta$ with nominal period $1/\xi$ pixels. Image ridges selected by that filter run perpendicular to $\theta$. The angle is relative to tile array axes, not celestial north.

## Filter definitions

Write $s=\texttt{slant}$ and let $\sigma_f$ and $\xi$ be in cycles per pixel. The **Gabor** filter is an elliptical Gaussian in Fourier space:

$$
\widehat\psi_{\rm G}(u,v)=
\exp\!\left[-\frac{(u_{\rm rot}-\xi)^2+(v_{\rm rot}/s)^2}{2\sigma_f^2}\right].
$$

It has a nonzero DC response. The **Morlet** filter subtracts a Gaussian centred at zero frequency:

$$
\widehat\psi_{\rm M}(u,v)=\widehat\psi_{\rm G}(u,v)
-K\exp\!\left[-\frac{u_{\rm rot}^2+(v_{\rm rot}/s)^2}{2\sigma_f^2}\right],
\qquad K=\exp\!\left[-\frac{\xi^2}{2\sigma_f^2}\right].
$$

Thus $\widehat\psi_{\rm M}(0,0)=0$. The subtraction can move the actual peak away from the nominal centre $\xi$.

For the **bump** filter, set $d^2=((u_{\rm rot}-\xi)^2+(v_{\rm rot}/s)^2)/\sigma_f^2$. Then

$$
\widehat\psi_{\rm B}(u,v)=
\begin{cases}
\exp\!\left(1+\dfrac{1}{d^2-1}\right),&d^2<1,\\
0,&d^2\geq1.
\end{cases}
$$

Its Fourier support is compact. It has zero DC response only when $\xi\geq\sigma_f$. The lowpass filter is isotropic:

$$
\widehat\phi_J(u,v)=\exp\!\left[-\frac{u^2+v^2}{2\sigma_{J,f}^2}\right],
\qquad \widehat\phi_J(0,0)=1.
$$

For Gabor and Morlet, the Fourier width parallel to the wavevector is $\sigma_f$ and the transverse width is $s\sigma_f$. When `slant < 1`, the Fourier response narrows transversely; its spatial envelope widens transversely. `slant` does not affect the lowpass.

## Filter-bank parameters and units

{py:func}`skykit.jax_wavelets.generate_filter_bank` accepts the following parameters:

| Parameter | Meaning in the current implementation |
| --- | --- |
| `M`, `N` | Tile dimensions in pixels. FFT-bin spacing is $1/M$ and $1/N$ cycles per pixel. |
| `J` | Number of wavelet scales $j=0,\ldots,J-1$; the lowpass uses scale $J$. |
| `L` | Number of orientations, $\theta_l=l\pi/L$ for $l=0,\ldots,L-1$. |
| `wavelet_type` | `"morlet"`, `"gabor"`, or `"bump"`. |
| `sigma0` | Base spatial scale in pixels. At scale $j$, $\sigma_j=\texttt{sigma0}\,2^j$ and $\sigma_{j,f}=1/(2\pi\sigma_j)$ cycles per pixel. For the Gaussian envelope, $\sigma_j$ is the spatial standard deviation along the wavevector. |
| `xi0` | Base carrier parameter. **As implemented**, $\xi_j=\texttt{xi0}/(\pi2^j)$ cycles per pixel, or angular frequency $2\,\texttt{xi0}/2^j$ radians per pixel. The input parameter is therefore half the scale-zero angular carrier frequency. |
| `slant` | Ratio of transverse to longitudinal Fourier width. The default `0.5` makes the transverse width half as large. |

The lowpass uses $\sigma_J=\texttt{sigma0}\,2^J$ and $\sigma_{J,f}=1/(2\pi\sigma_J)$. With the current default `xi0=π/4`, the nominal scale-zero carrier is $0.25$ cycles per pixel, a four-pixel period. `xi0` is not interchangeable with Kymatio's radians-per-pixel carrier parameter; equal numerical values give different filters.

## Scattering paths and outputs

For periodic convolution $*$ on a tile, the implemented paths are

$$
U_1(j_1,\theta_1)x=x*\psi_{j_1,\theta_1}.
$$

$$
U_2(j_1,\theta_1,j_2,\theta_2)x
=|U_1(j_1,\theta_1)x|*\psi_{j_2,\theta_2},
\qquad j_2>j_1.
$$

{py:func}`skykit.wavelet_transform.wavelet_transform_tile` returns a {py:class}`skykit.feature_maps.FeatureMap` for one tile or a {py:class}`skykit.feature_maps.FeatureMapSet` for a TileSet. `order=1` computes `U1`; `order=2` adds `U2`. Both are complex wavelet responses: the modulus in `U2` is applied to `U1` before the second convolution, and no modulus is applied afterward. Each object's `paths` maps a feature index to `(j, theta)` or `(j1, theta1, j2, theta2)`. `feature(group, path, ...)` fetches a map by its wavelet parameters. The generated lowpass filter is not used in this feature stage.

The FFT implements circular convolution. Apodizing the tile margin reduces boundary artifacts but does not make planar pixel frequencies identical to angular frequencies on the sphere. For `pol=True`, Q and U are transformed separately as scalar arrays; the nonlinear feature channels should not be interpreted as new Stokes Q/U fields.

## Statistics from feature maps

{py:func}`skykit.scattering_transform.scattering_transform` accepts a `FeatureMap`, `FeatureMapSet`, or HDF5 path. It returns a {py:class}`skykit.scattering_transform.ScatteringStatistics` object. Its `values` and `paths` dictionaries share keys. A single input gives one statistic per tile and wavelet path. Two inputs give only cross statistics for all path pairs; `U1_U2`, for example, has shape `(tiles, paths_in_set_1, paths_in_set_2)`. Polarization adds an axis after tiles. For a single tile the tile axis is absent. `statistic(group, path1, path2, tile=..., stokes=...)` retrieves a value using wavelet parameters.

For one set, the value is the pixel mean or population variance of `operation1(U)`. For two sets, the value is the pixel mean or population variance of `operation1(U) * operation2(V)`. Both operations are optional JAX functions and must preserve map shape. Their default is identity, so a mean can be complex. Pass `operation1=jax.numpy.abs` and, for cross statistics, `operation2=jax.numpy.abs` to reduce magnitudes. The full tile, including margins, participates in each reduction.

For large sets, use `wavelet_transform_tile(tiles, filters, order=2, filepath="feature_maps.h5", batch_size=4)`. This writes the `U1` and `U2` datasets and their path arrays into one HDF5 file while transforming small tile batches. A single tile uses the same `filepath` option. The returned feature object reads maps lazily; close it with a `with` block or `.close()`. `FeatureMap.to_hdf5` and `FeatureMapSet.to_hdf5` save already computed maps. `open_feature_maps(path)` reopens either type. The statistics function reads a tile and a limited number of paths at a time; `path_batch_size` controls the path chunk size.
