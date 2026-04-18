import numpy as np
import jax.numpy as jnp

def _compute_grid(M, N):
    """Generate centred frequency grid (-0.5 to 0.5) for an M x N image."""
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
        Spatial dimensions of the patch.
    sigma : float
        Bandwidth parameter.
    theta : float
        Orientation angle in radians.
    xi : float
        Central frequency.
    slant : float
        Slant (aspect ratio) of the envelope.
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
        Spatial dimensions.
    sigma : float
        Bandwidth parameter.
    theta : float
        Orientation angle in radians.
    xi : float
        Central frequency.
    slant : float
        Slant (aspect ratio) of the envelope.
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
        Spatial dimensions.
    sigma : float
        Bandwidth parameter. Controls the cutoff frequency of the lowpass.
    """
    u, v = _compute_grid(M, N)
    arg = (u**2 + v**2) / (2 * sigma**2)
    return jnp.exp(-arg)

def bump_2d(M, N, sigma, theta, xi, slant=0.5):
    """
    Computes a 2D Bump wavelet in the Fourier domain.
    Bump wavelets have strictly compact support in the frequency domain.
    """
    u, v = _compute_grid(M, N)
    
    u_rot = u * jnp.cos(theta) + v * jnp.sin(theta)
    v_rot = -u * jnp.sin(theta) + v * jnp.cos(theta)
    
    dist = ((u_rot - xi)**2 + (v_rot / slant)**2) / sigma**2
    # Bump function: exp(1 / (dist^2 - 1)) if dist < 1 else 0
    mask = dist < 1.0
    # Add a small epsilon to avoid division by zero in the invalid region 
    # (mask will zero it out anyway)
    dist_safe = jnp.where(mask, dist, 0.0)
    val = jnp.exp(1.0 / (dist_safe**2 - 1.0))
    # Normalize to peak at 1
    val = val * jnp.exp(1.0)
    return jnp.where(mask, val, 0.0)

def generate_filter_bank(M, N, J, L, wavelet_type='morlet', sigma0=0.8, xi0=np.pi/4.0, slant=0.5):
    """
    Generate a complete filter bank (wavelets and low-pass) for 2D scattering.
    
    Parameters
    ----------
    M, N : int
        Spatial dimensions.
    J : int
        Maximum scale.
    L : int
        Number of orientations.
    wavelet_type : str
        'morlet', 'gabor', or 'bump'.
    sigma0 : float
        Base bandwidth.
    xi0 : float
        Base centre frequency.
    slant : float
        Slant (aspect ratio).
        
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
    else:
        raise ValueError(f"Unknown wavelet_type: {wavelet_type}")
        
    # Generate wavelets for each scale and orientation
    for j in range(J):
        # Scale the frequency and bandwidth
        sigma_j = sigma0 * (2 ** j)
        # Using a normalized frequency grid [-0.5, 0.5], so we divide xi by 2^j but also scale it to match FFT frequencies
        # Standard Kymatio definitions often base xi on pixel units
        xi_j = (xi0 / np.pi) / (2 ** j)
        # sigma in frequency coords is 1 / (spatial sigma)
        sigma_f = 1.0 / (sigma_j * 2 * np.pi)
        
        for l in range(L):
            theta = l * np.pi / L
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
