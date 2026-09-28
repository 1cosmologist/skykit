"""Fourier-domain filters and dyadic banks for planar tile scattering.

Frequencies use unshifted FFT bins in cycles per tile pixel. Filter angles
refer to tile array axes, not celestial directions.
"""

import numpy as np
import jax.numpy as jnp
from math import comb, sqrt

def _compute_grid(M, N):
    """Generate the unshifted FFT frequency grid for an M by N tile."""
    # jnp.fft.fftfreq returns [0, 1/M, ..., 0.5, -0.5+1/M, ..., -1/M]
    # We want the unshifted grid to apply functions, then we can shift or keep as is.
    # It's usually easiest to evaluate directly on the fftfreq grid.
    fx = jnp.fft.fftfreq(M)
    fy = jnp.fft.fftfreq(N)
    u, v = jnp.meshgrid(fx, fy, indexing='ij')
    return u, v

def gabor_2d(M, N, sigma, theta, xi, slant=0.5):
    """
    Computes a 2D Gabor filter in the Fourier domain.
    
    Parameters
    ----------
    M, N : int
        Spatial dimensions of the tile in pixels.
    sigma : float
        Fourier standard deviation along the wavevector, in cycles per pixel.
    theta : float
        Wavevector angle from the first tile-array axis, in radians.
    xi : float
        Carrier frequency in cycles per pixel.
    slant : float
        Ratio of transverse to longitudinal Fourier width.
    """
    u, v = _compute_grid(M, N)
    
    # Rotate the grid by -theta
    u_rot = u * jnp.cos(theta) + v * jnp.sin(theta)
    v_rot = -u * jnp.sin(theta) + v * jnp.cos(theta)
    
    # Gabor in Fourier space
    # exp( - ( (u_rot - xi)^2 + (v_rot / slant)^2 ) / (2 * sigma^2) )
    arg = ((u_rot - xi)**2 + (v_rot / slant)**2) / (2 * sigma**2)
    return jnp.exp(-arg)

def morlet_2d(M, N, sigma, theta, xi, slant=0.5):
    """
    Computes a 2D Morlet wavelet in the Fourier domain.
    The Morlet wavelet is a Gabor filter corrected to have exactly zero mean.
    
    Parameters
    ----------
    M, N : int
        Spatial dimensions of the tile in pixels.
    sigma : float
        Fourier width along the wavevector, in cycles per pixel.
    theta : float
        Wavevector angle from the first tile-array axis, in radians.
    xi : float
        Nominal carrier frequency in cycles per pixel; the zero-mean
        correction may move the actual filter peak.
    slant : float
        Ratio of transverse to longitudinal Fourier width.
    """
    # Gabor term
    gabor = gabor_2d(M, N, sigma, theta, xi, slant=slant)
    
    u, v = _compute_grid(M, N)
    u_rot = u * jnp.cos(theta) + v * jnp.sin(theta)
    v_rot = -u * jnp.sin(theta) + v * jnp.cos(theta)
    
    # Low pass term (to subtract the mean)
    arg_lp = (u_rot**2 + (v_rot / slant)**2) / (2 * sigma**2)
    low_pass = jnp.exp(-arg_lp)
    
    # The correction weight K ensures that at (u=0, v=0), the filter is 0
    K = jnp.exp(-(xi**2) / (2 * sigma**2))
    
    return gabor - K * low_pass

def lowpass_2d(M, N, sigma):
    """
    Computes a 2D Gaussian lowpass filter (phi) in the Fourier domain.
    
    Parameters
    ----------
    M, N : int
        Spatial dimensions of the tile in pixels.
    sigma : float
        Gaussian Fourier standard deviation in cycles per pixel.
    """
    u, v = _compute_grid(M, N)
    arg = (u**2 + v**2) / (2 * sigma**2)
    return jnp.exp(-arg)

def bump_2d(M, N, sigma, theta, xi, slant=0.5):
    """
    Computes a 2D analytic bump wavelet in the Fourier domain.

    The filter has strictly compact support — it is the smooth bump function
    ``exp(1/(d²-1)) * 1[d²<1]`` evaluated on an elliptical disk centred at
    ``(xi, 0)`` in the rotated coordinate frame, where ``d²`` is the
    normalised squared distance from the centre.

    When ``xi >= sigma``, the support stays on the positive-frequency side
    and excludes DC. The spatial filter is then complex and directional.
    For ``xi < sigma``, the support includes DC, so it is not zero-mean.

    It is not the half-plane angular bump-steerable filter implemented by
    ``bump_steerable_2d``. Its rotated elliptical supports do not form a
    finite steering basis, even when this filter has an analytic response.

    The compact Fourier support avoids inter-scale aliasing better than
    Gaussian-tailed wavelets on finite grids.
    """
    u, v = _compute_grid(M, N)
    
    u_rot = u * jnp.cos(theta) + v * jnp.sin(theta)
    v_rot = -u * jnp.sin(theta) + v * jnp.cos(theta)
    
    dist = ((u_rot - xi)**2 + (v_rot / slant)**2) / sigma**2
    # dist = d^2 (squared normalised distance). Bump: exp(1 / (d^2 - 1)) for d^2 < 1
    mask = dist < 1.0
    # Replace out-of-support values with 0 to avoid division by zero
    dist_safe = jnp.where(mask, dist, 0.0)
    val = jnp.exp(1.0 / (dist_safe - 1.0))
    # Normalize to peak at 1
    val = val * jnp.exp(1.0)
    return jnp.where(mask, val, 0.0)

def bump_steerable_2d(M, N, j, theta, L, xi0=0.45 * np.pi):
    """Fourier-domain analytic bump with a half-plane angular profile.

    The radial bump peaks at ``xi0 / 2**j`` radians per pixel and is supported
    on ``0 < |k| < 2 * xi0 / 2**j``. The angular factor is
    ``max(cos(arg(k) - theta), 0)**(L-1)``. Its positive-frequency support
    gives a complex spatial response. The normalization covers the angular
    energy of the L filters and their conjugate antipodal orientations; it
    does not normalize the complete multiscale filter bank.

    Parameters
    ----------
    M, N : int
        Tile dimensions.
    j : int
        Dyadic scale, starting at zero.
    theta : float
        Wavevector direction from the first tile axis, in radians.
    L : int
        Number of directions in ``[0, pi)``; at least two.
    xi0 : float
        Scale-zero radial peak in radians per pixel. Must be in ``(0, pi/2)``
        so the finest-scale support stays inside the Nyquist disk.

    Notes
    -----
    The half-plane cutoff makes the whole complex filter only approximately
    steerable from finitely many orientations. For odd L its real spatial
    component is exactly steerable; for even L its imaginary component is.
    """
    if not isinstance(L, (int, np.integer)) or L < 2:
        raise ValueError("L must be an integer >= 2")
    if not isinstance(j, (int, np.integer)) or j < 0:
        raise ValueError("j must be a nonnegative integer")
    if not np.isfinite(xi0) or not 0 < xi0 < np.pi / 2:
        raise ValueError("xi0 must be in (0, pi/2) radians per pixel")

    u, v = _compute_grid(M, N)
    kx, ky = 2 * jnp.pi * u, 2 * jnp.pi * v
    r = jnp.hypot(kx, ky)
    xi = xi0 * 2.0 ** (-j)

    t = (r - xi) / xi
    inside = jnp.abs(t) < 1.0
    safe_t = jnp.where(inside, t, 0.0)
    radial = jnp.where(inside, jnp.exp(-safe_t**2 / (1.0 - safe_t**2)), 0.0)

    r_safe = jnp.where(r > 0, r, 1.0)
    directional_cosine = (kx * jnp.cos(theta) + ky * jnp.sin(theta)) / r_safe
    angular = jnp.maximum(jnp.clip(directional_cosine, -1.0, 1.0), 0.0) ** (L - 1)
    alpha = sqrt(4 ** (L - 1) / (L * comb(2 * (L - 1), L - 1)))
    return alpha * radial * angular


def generate_filter_bank(M, N, J, L, wavelet_type='morlet', sigma0=0.8,
                         xi0=np.pi/4.0, slant=0.5,
                         bump_steerable_xi0=0.45 * np.pi):
    """
    Generate a complete filter bank (wavelets and low-pass) for 2D scattering.
    
    Parameters
    ----------
    M, N : int
        Spatial dimensions of the tile in pixels.
    J : int
        Number of wavelet scales, indexed from 0 to J-1. The lowpass is at J.
    L : int
        Number of wavevector directions spanning [0, pi), relative to the
        first tile-array axis.
    wavelet_type : str
        'morlet', 'gabor', 'bump', or 'bump_steerable'.
    sigma0 : float
        Base spatial scale in pixels. The Gaussian envelope has this spatial
        standard deviation along its wavevector at scale zero. For
        'bump_steerable', it controls only the lowpass filter.
    xi0 : float
        Base carrier parameter for 'morlet', 'gabor', and 'bump'. The nominal scale-zero
        frequency is ``xi0 / pi`` cycles per pixel, equivalent to
        ``2 * xi0`` radians per pixel. Not used by 'bump_steerable'.
    slant : float
        Ratio of transverse to longitudinal Fourier width.
        Not used by 'bump_steerable'.
    bump_steerable_xi0 : float
        Scale-zero radial peak for 'bump_steerable', in radians per pixel.
        Kept separate from ``xi0``, which has different units and semantics.
        
    Returns
    -------
    filters : dict
        Contains:
        - 'psi': list of dicts, each with 'j', 'theta', and 'val' (the 2D Fourier array)
        - 'phi': dict with 'j' and 'val' (the 2D Fourier array)
    """
    filters = {'psi': [], 'phi': None}
    
    if wavelet_type == 'morlet':
        wav_func = morlet_2d
    elif wavelet_type == 'gabor':
        wav_func = gabor_2d
    elif wavelet_type == 'bump':
        wav_func = bump_2d
    elif wavelet_type == 'bump_steerable':
        wav_func = None
    else:
        raise ValueError(f"Unknown wavelet_type: {wavelet_type}")
        
    # Generate wavelets for each scale and orientation
    for j in range(J):
        if wavelet_type != 'bump_steerable':
            sigma_j = sigma0 * (2 ** j)
            xi_j = (xi0 / np.pi) / (2 ** j)
            sigma_f = 1.0 / (sigma_j * 2 * np.pi)
        
        for l in range(L):
            theta = l * np.pi / L
            if wavelet_type == 'bump_steerable':
                f_val = bump_steerable_2d(
                    M, N, j, theta, L, xi0=bump_steerable_xi0)
            else:
                f_val = wav_func(M, N, sigma_f, theta, xi_j, slant=slant)
            filters['psi'].append({
                'j': j,
                'theta': theta,
                'val': f_val
            })
            
    # Generate the low-pass filter (phi) at the maximum scale J
    sigma_J = sigma0 * (2 ** J)
    sigma_J_f = 1.0 / (sigma_J * 2 * np.pi)
    filters['phi'] = {
        'j': J,
        'val': lowpass_2d(M, N, sigma_J_f)
    }
    
    return filters
