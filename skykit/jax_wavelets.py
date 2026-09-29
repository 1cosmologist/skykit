"""Fourier-domain filters and dyadic banks for planar tile scattering.

Frequencies use unshifted FFT bins in cycles per tile pixel. Filter angles
refer to tile array axes, not celestial directions.
"""

import math
import warnings

import numpy as np
import jax.numpy as jnp
from jax.scipy.special import factorial

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

def bump_steerable_2d(M, N, j, theta, L, xi0=0.45 * np.pi,
                      scales_per_octave=1):
    """Fourier-domain analytic bump with a half-plane angular profile.

    The radial bump peaks at ``xi0 / 2**(j / scales_per_octave)`` radians per
    pixel and is supported on ``0 < |k| < 2 * xi0 / 2**(j / scales_per_octave)``.
    The angular factor is
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
    scales_per_octave : int
        Number of scales per octave; defaults to dyadic spacing.

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
    if (not isinstance(scales_per_octave, (int, np.integer))
            or isinstance(scales_per_octave, (bool, np.bool_))
            or scales_per_octave < 1):
        raise ValueError("scales_per_octave must be a positive integer")

    u, v = _compute_grid(M, N)
    kx, ky = 2 * jnp.pi * u, 2 * jnp.pi * v
    r = jnp.hypot(kx, ky)
    xi = xi0 * 2.0 ** (-j / scales_per_octave)

    t = (r - xi) / xi
    inside = jnp.abs(t) < 1.0
    safe_t = jnp.where(inside, t, 0.0)
    radial = jnp.where(inside, jnp.exp(-safe_t**2 / (1.0 - safe_t**2)), 0.0)

    r_safe = jnp.where(r > 0, r, 1.0)
    directional_cosine = (kx * jnp.cos(theta) + ky * jnp.sin(theta)) / r_safe
    angular = jnp.maximum(jnp.clip(directional_cosine, -1.0, 1.0), 0.0) ** (L - 1)
    angular_order = L - 1
    alpha = (2.0 ** angular_order * factorial(angular_order)
             / jnp.sqrt(L * factorial(2 * angular_order)))
    return alpha * radial * angular


def suggest_filter_bank_params(beam_fwhm_arcmin, patch_shape,
                               pixel_size_arcmin, wavelet_type='morlet', *,
                               L=8, xi_scale=1.0, J=None,
                               scales_per_octave=None, sigma_xi=None,
                               width_ratio=0.7, angular_coverage=1.1,
                               slant=None):
    """Suggest keyword arguments for :func:`generate_filter_bank`.

    The finest nominal carrier has a half-period equal to the beam FWHM,
    multiplied by ``xi_scale``. This is a heuristic: the beam transfer
    function at that frequency is about 0.41 when ``xi_scale=1``.
    Frequencies are limited to the radial Nyquist disk. For Gaussian filters
    the chosen limit is carrier plus two longitudinal Fourier standard
    deviations; Gaussian tails extend beyond that limit.

    Parameters
    ----------
    beam_fwhm_arcmin : float
        Map beam full width at half maximum in arcminutes.
    patch_shape : tuple of int
        Tile shape ``(M, N)`` in pixels.
    pixel_size_arcmin : float
        Pixel width in arcminutes; assumes square pixels.
    wavelet_type : str
        ``'morlet'``, ``'gabor'``, ``'bump'``, or ``'bump_steerable'``.
    L : int
        Number of stored orientations in ``[0, pi)``.
    xi_scale : float
        Multiplier for the beam-derived finest carrier.
    J : int or None
        Number of scales. ``None`` chooses the largest count for which the
        coarsest carrier has at least ``4*L`` samples around its FFT ring.
    scales_per_octave : int or None
        Defaults to 2 for Gabor and 1 for the other filters.
    sigma_xi : float or None
        Product of Gaussian spatial width (pixels) and carrier (rad/pixel).
        Defaults to ``0.6*pi`` for Morlet and 4 for Gabor.
    width_ratio : float
        Bump Fourier width divided by carrier; must be in ``(1/3, 1)``.
    angular_coverage : float
        Desired bump support half-angle in units of ``pi/L``. The slant is
        capped at 1 if this coverage is geometrically impossible.
    slant : float or None
        Explicit transverse/longitudinal Fourier width ratio in ``(0, 1]``.

    Returns
    -------
    dict
        Arguments accepted directly by ``generate_filter_bank``. In its
        interface, ``xi0`` is *half* the carrier in radians per pixel;
        ``bump_steerable_xi0`` is the carrier itself. ``sigma0`` is a spatial
        width in pixels, including for the elliptical bump.
    """
    def positive(name, value):
        try:
            valid = math.isfinite(value) and value > 0
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError(f"{name} must be finite and positive")

    positive('beam_fwhm_arcmin', beam_fwhm_arcmin)
    positive('pixel_size_arcmin', pixel_size_arcmin)
    positive('xi_scale', xi_scale)
    if (not isinstance(patch_shape, (tuple, list)) or len(patch_shape) != 2
            or any(not isinstance(n, (int, np.integer))
                   or isinstance(n, (bool, np.bool_)) or n <= 0
                   for n in patch_shape)):
        raise ValueError('patch_shape must contain two positive integers')
    M, N = map(int, patch_shape)
    if (not isinstance(L, (int, np.integer))
            or isinstance(L, (bool, np.bool_)) or L < 2):
        raise ValueError('L must be an integer >= 2')
    L = int(L)
    if not isinstance(wavelet_type, str):
        raise ValueError('wavelet_type must be a string')
    wavelet_type = wavelet_type.lower()
    if wavelet_type not in ('morlet', 'gabor', 'bump', 'bump_steerable'):
        raise ValueError(f'Unknown wavelet_type: {wavelet_type}')
    if scales_per_octave is None:
        scales_per_octave = 2 if wavelet_type == 'gabor' else 1
    if (not isinstance(scales_per_octave, (int, np.integer))
            or isinstance(scales_per_octave, (bool, np.bool_))
            or scales_per_octave < 1):
        raise ValueError('scales_per_octave must be a positive integer')
    Q = int(scales_per_octave)
    if J is not None and (not isinstance(J, (int, np.integer))
                          or isinstance(J, (bool, np.bool_)) or J < 1):
        raise ValueError('J must be a positive integer')
    if slant is not None:
        positive('slant', slant)
        if slant > 1:
            raise ValueError('slant must be <= 1')

    xi_fine = xi_scale * math.pi * pixel_size_arcmin / beam_fwhm_arcmin
    if wavelet_type in ('morlet', 'gabor'):
        if sigma_xi is None:
            sigma_xi = 0.6 * math.pi if wavelet_type == 'morlet' else 4.0
        positive('sigma_xi', sigma_xi)
        xi_limit = math.pi / (1.0 + 2.0 / sigma_xi)
    elif wavelet_type == 'bump':
        positive('width_ratio', width_ratio)
        if not 1.0 / 3.0 < width_ratio < 1.0:
            raise ValueError('width_ratio must be in (1/3, 1)')
        positive('angular_coverage', angular_coverage)
        xi_limit = math.pi / (1.0 + width_ratio)
    else:
        # Leave room below the strict xi0 < pi/2 condition of the filter.
        xi_limit = 0.45 * math.pi

    if xi_fine > xi_limit:
        warnings.warn(
            f'Beam-derived carrier {xi_fine:.3f} rad/pixel exceeds the '
            f'{wavelet_type} Nyquist limit; using {xi_limit:.3f} rad/pixel.',
            UserWarning, stacklevel=2)
        xi_fine = xi_limit

    ring_ratio = xi_fine * min(M, N) / (4.0 * L)
    if ring_ratio < 1.0:
        raise ValueError('Patch too small to resolve L orientations at the '
                         'finest scale; reduce L or enlarge the patch')
    max_J = math.floor(Q * math.log2(ring_ratio) + 1e-12) + 1
    if J is None:
        J = max_J
    elif J > max_J:
        raise ValueError(f'J={J} exceeds the geometry-based maximum {max_J}')

    params = dict(M=M, N=N, J=int(J), L=L, wavelet_type=wavelet_type,
                  scales_per_octave=Q)
    if wavelet_type in ('morlet', 'gabor'):
        params['sigma0'] = sigma_xi / xi_fine
        params['xi0'] = xi_fine / 2.0
        params['slant'] = (float(slant) if slant is not None else
                           min(1.0, (4.0 / L) * sigma_xi / (0.6 * math.pi)))
    elif wavelet_type == 'bump':
        params['sigma0'] = 1.0 / (width_ratio * xi_fine)
        params['xi0'] = xi_fine / 2.0
        half_angle = min(angular_coverage * math.pi / L,
                         math.pi / 2.0 - 1e-9)
        params['slant'] = (float(slant) if slant is not None else
                           min(1.0, math.sqrt(1.0 - width_ratio**2)
                               * math.tan(half_angle) / width_ratio))
    else:
        params['sigma0'] = 1.0 / xi_fine  # sets the bank's Gaussian lowpass
        params['bump_steerable_xi0'] = xi_fine
    return params


def generate_filter_bank(M, N, J, L, wavelet_type='morlet', sigma0=0.8,
                         xi0=np.pi/4.0, slant=0.5,
                         bump_steerable_xi0=0.45 * np.pi,
                         scales_per_octave=1):
    """
    Generate a complete filter bank (wavelets and low-pass) for 2D scattering.
    
    Parameters
    ----------
    M, N : int
        Spatial dimensions of the tile in pixels.
    J : int
        Number of wavelet scales, indexed from 0 to J-1. The lowpass is at J.
        Scale frequencies fall by ``2**(1 / scales_per_octave)`` per step.
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
    scales_per_octave : int
        Number of scales per octave. Defaults to 1 (dyadic spacing).
        
    Returns
    -------
    filters : dict
        Contains:
        - 'psi': list of dicts, each with 'j', 'theta', and 'val' (the 2D Fourier array)
        - 'phi': dict with 'j' and 'val' (the 2D Fourier array)
    """
    if (not isinstance(scales_per_octave, (int, np.integer))
            or isinstance(scales_per_octave, (bool, np.bool_))
            or scales_per_octave < 1):
        raise ValueError("scales_per_octave must be a positive integer")
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
            sigma_j = sigma0 * (2 ** (j / scales_per_octave))
            xi_j = (xi0 / np.pi) / (2 ** (j / scales_per_octave))
            sigma_f = 1.0 / (sigma_j * 2 * np.pi)
        
        for l in range(L):
            theta = l * np.pi / L
            if wavelet_type == 'bump_steerable':
                f_val = bump_steerable_2d(
                    M, N, j, theta, L, xi0=bump_steerable_xi0,
                    scales_per_octave=scales_per_octave)
            else:
                f_val = wav_func(M, N, sigma_f, theta, xi_j, slant=slant)
            filters['psi'].append({
                'j': j,
                'theta': theta,
                'val': f_val
            })
            
    # Generate the low-pass filter (phi) at the maximum scale J
    sigma_J = sigma0 * (2 ** (J / scales_per_octave))
    sigma_J_f = 1.0 / (sigma_J * 2 * np.pi)
    filters['phi'] = {
        'j': J,
        'val': lowpass_2d(M, N, sigma_J_f)
    }
    
    return filters
