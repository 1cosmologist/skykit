# Mathematics of Skykit Wavelets and Scattering Transform

This document details the mathematical formulation of the wavelets and the scattering transform implemented in `skykit`. The implementations are exacted in Fourier space to allow for fast, alias-free convolutions. 

Functions referenced here are located in `skykit/jax_wavelets.py` and `skykit/scattering_transform.py`.

---

## 1. Frequency Grid and Coordinates

All filters are generated directly in the 2D frequency domain using normalized spatial frequencies $(u, v) \in [-0.5, 0.5)$. For an $M \times N$ image, `_compute_grid(M, N)` creates this centered representation.

To apply orientation, the frequency grid is rotated by an angle $\theta$ (in radians):

$$ u_{\text{rot}} = u \cos\theta + v \sin\theta $$

$$ v_{\text{rot}} = -u \sin\theta + v \cos\theta $$

The aspect ratio of the filters transversally to their main directional propagation is controlled by the `slant` parameter.

---

## 2. Wavelet Definitions (`jax_wavelets.py`)

### 2.1. Gabor Wavelet (`gabor_2d`)
The Gabor filter is a complex sinusoid modulated by a Gaussian envelope. In the Fourier domain, it is simply a shifted Gaussian:

$$ \hat{\psi}_{\text{Gabor}}(u, v) = \exp\left( - \frac{(u_{\text{rot}} - \xi)^2 + (v_{\text{rot}} / \text{slant})^2}{2 \sigma_f^2} \right) $$

**Parameters:**
*   `sigma`: $\sigma_f$, the frequency-domain bandwidth.
*   `xi`: $\xi$, the central frequency of the wavelet.

### 2.2. Morlet Wavelet (`morlet_2d`)
A pure Gabor filter has a small non-zero mean. The Morlet wavelet corrects this by subtracting a lowpass term, guaranteeing the admissibility condition $\hat{\psi}(0, 0) = 0$:

$$ \hat{\psi}_{\text{Morlet}}(u, v) = \hat{\psi}_{\text{Gabor}}(u, v) - K \exp\left( - \frac{u_{\text{rot}}^2 + (v_{\text{rot}} / \text{slant})^2}{2 \sigma_f^2} \right) $$

Where $K = \exp\left(-\frac{\xi^2}{2 \sigma_f^2}\right)$ is the correction weight evaluated at the origin.

### 2.3. Bump Wavelet (`bump_2d`)
The Bump wavelet has strictly compact support in the frequency domain, meaning it is exactly zero outside a defined radius, eliminating cross-talk between distant frequency bands. Let the squared scaled distance be:

$$ d^2 = \frac{(u_{\text{rot}} - \xi)^2 + (v_{\text{rot}} / \text{slant})^2}{\sigma_f^2} $$

Then:

$$ \hat{\psi}_{\text{Bump}}(u, v) = \begin{cases} e \cdot \exp\left(\frac{1}{d^2 - 1}\right) & \text{if } d^2 < 1 \\ 0 & \text{otherwise} \end{cases} $$

Scaled by Euler's number $e$ so the peak value is 1.

### 2.4. Lowpass Filter (`lowpass_2d`)
The scaling function $\phi$ describes the invariant spatial averaging of the transform.

$$ \hat{\phi}(u, v) = \exp\left( - \frac{u^2 + v^2}{2 \sigma_f^2} \right) $$

---

## 3. Filter Bank Scaling (`generate_filter_bank`)

The filter bank consists of wavelets generated at discrete dyadic scales $j \in \{0, \dots, J-1\}$ and orientations $l \in \{0, \dots, L-1\}$.

Given base parameters `sigma0` ($\sigma_0$) and `xi0` ($\xi_0$), the properties of the $j$-th scale are dilated:
*   **Spatial Bandwidth:** $\sigma_j = \sigma_0 \cdot 2^j$
*   **Frequency Bandwidth:** $\sigma_f = \frac{1}{2\pi \sigma_j}$
*   **Central Frequency:** $\xi_j = \frac{\xi_0 / \pi}{2^j}$
*   **Angle:** $\theta_l = l \cdot \frac{\pi}{L}$

The lowpass filter $\phi$ corresponds to the maximum scale $J$, smoothing over the largest spatial extent: $\sigma_J = \sigma_0 \cdot 2^J$.

---

## 4. The 2D Scattering Transform (`Scattering2D`)

The `Scattering2D` class implements an undecimated (no spatial downsampling) transform, operating in batches over tensors $x$. By default, the lowpass-averaged coefficients retain full spatial resolution for tile-aligned analysis.

Let $x \ast \phi$ denote spatial convolution. In `skykit`, this is evaluated via the FFT:

$$ x \ast \psi = \mathcal{F}^{-1} \left( \mathcal{F}(x) \cdot \hat{\psi} \right) $$

### 4.1. Zeroth-Order ($S_0$)
Retrieves the locally translation-invariant component (the smoothed background).

$$ S_0 x = x \ast \phi $$

Implemented as `jnp.real(ifft2(fft2(x) * phi_val))`.

### 4.2. First-Order ($S_1$)
Extracts directional gradient/edge features at scale $j_1$ and angle $\theta_1$.
1. **Modulus (non-linear operator):** $U_1(j_1, \theta_1) x = |x \ast \psi_{j_1, \theta_1}|$
2. **Averaging:** $S_1(j_1, \theta_1) x = U_1(j_1, \theta_1) x \ast \phi$

Implemented as a batched product over the full stacked $\psi$ tensor, then averaged with $\phi$.

### 4.3. Second-Order ($S_2$)
Extracts interference patterns (e.g., textures, branching filaments) between different scales. To prevent exponential energy blow-up, interference paths only cascade from higher to lower spatial frequencies (i.e. $j_2 > j_1$).
1.  **Modulus:** $U_2(j_1, \theta_1, j_2, \theta_2) x = | U_1(j_1, \theta_1) x \ast \psi_{j_2, \theta_2} |$
2.  **Averaging:** $S_2(j_1, \theta_1, j_2, \theta_2) x = U_2(j_1, \theta_1, j_2, \theta_2) x \ast \phi$

The default result from `transform_tile()` or `transform_tileset()` contains maps `S0`, `S1`, and `S2`, with channels mapping to paths sequentially. With `spatial_average=True`, each map is reduced to its arithmetic mean over the entire tile, including margins. With `return_feature_maps=True`, the result additionally contains `U1` and `U2`: the modulus maps before convolution with $\phi_J$. These remain maps even when `spatial_average=True`. Because $\hat\phi_J(0)=1$ and the FFT convolution is periodic, the spatial mean of each `S` map equals that of its corresponding unaveraged `U` map, up to numerical precision.
