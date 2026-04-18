import jax
import jax.numpy as jnp
import numpy as np
from .jax_wavelets import generate_filter_bank

class Scattering2D:
    """
    A 2D Wavelet Scattering Transform, defined using JAX for fast execution on GPUs.
    Computes scattering coefficients directly on pre-tiled batches (from a TileSet).
    Uses undecimated formulation (no spatial downsampling) to preserve 1:1 pixel 
    correspondence with the original tiles, allowing easy reconstruction or analysis.
    """
    
    def __init__(self, M, N, J=3, L=8, wavelet_type='morlet', sigma0=0.8, xi0=0.785, max_order=2):
        """
        Initialize the Scattering Transform filter bank.
        
        Parameters
        ----------
        M, N : int
            Spatial dimensions of the input tiles.
        J : int
            Maximum scale of the wavelets.
        L : int
            Number of orientations per scale.
        wavelet_type : str
            Type of wavelet ('morlet', 'gabor', 'bump').
        sigma0 : float
            Base spatial bandwidth.
        xi0 : float
            Base center frequency.
        max_order : int
            Max scattering order (1 or 2).
        """
        self.M = M
        self.N = N
        self.J = J
        self.L = L
        self.max_order = max_order
        
        # Generate the JAX-compatible filter bank in Fourier space
        self.filters = generate_filter_bank(M, N, J, L, wavelet_type=wavelet_type, 
                                            sigma0=sigma0, xi0=xi0)
        
        # Precompile and format the filter arrays for vectorised application
        # Stack all psi filters into a single tensor of shape (N_psi, M, N)
        self.psi_vals = jnp.stack([f['val'] for f in self.filters['psi']])
        # Ensure primitive Python ints are used so jax.jit doesn't trace the mask indices
        self.psi_j = np.array([int(f['j']) for f in self.filters['psi']])
        
        # Lowpass filter phi shape (M, N)
        self.phi_val = self.filters['phi']['val']
        
        # JIT compile the batched transform function
        self._transform_jit = jax.jit(self._compute_coefficients)

    def _compute_coefficients(self, x):
        """
        Internal JAX function to compute scattering paths on a batch.
        
        Parameters
        ----------
        x : array-like (B, M, N)
            Batch of input patches.
            
        Returns
        -------
        out : dict
            'S0': shape (B, M, N)
            'S1': shape (B, N_psi, M, N)
            'S2': shape (B, N_paths_order2, M, N)
        """
        # FFT of the input: (B, M, N)
        x_f = jnp.fft.fft2(x)
        
        # Order 0: x * phi
        S0_f = x_f * self.phi_val[jnp.newaxis, :, :]
        S0 = jnp.real(jnp.fft.ifft2(S0_f))
        
        out = {'S0': S0}
        
        if self.max_order >= 1:
            # Order 1: |x * psi1|
            # x_f (B, 1, M, N) * psi_vals (1, N_psi, M, N) -> (B, N_psi, M, N)
            U1_f = x_f[:, jnp.newaxis, :, :] * self.psi_vals[jnp.newaxis, :, :, :]
            U1 = jnp.abs(jnp.fft.ifft2(U1_f))
            
            # S1 = U1 * phi
            U1_f_new = jnp.fft.fft2(U1)
            S1_f = U1_f_new * self.phi_val[jnp.newaxis, jnp.newaxis, :, :]
            S1 = jnp.real(jnp.fft.ifft2(S1_f))
            out['S1'] = S1
            
            if self.max_order >= 2:
                # Order 2: ||x * psi1| * psi2|
                # We only compute paths where j2 > j1 (frequency decreasing path to avoid energy explosion)
                S2_list = []
                # Looping over j1 manually in JAX trace (fully unrolled)
                for i1 in range(len(self.psi_vals)):
                    # Extract U1_f for path i1: shape (B, M, N)
                    u1_f_curr = U1_f_new[:, i1, :, :]
                    
                    # Valid j2 range is j2 > j1
                    valid_mask = self.psi_j > self.psi_j[i1]
                    
                    if not np.any(valid_mask):
                        continue
                        
                    # Filter self.psi_vals directly for this mask
                    # (In JAX, dynamic shapes inside tracing are tricky, but since the mask is static (only depends on J), we can use boolean indexing)
                    psi_j2_vals = self.psi_vals[valid_mask]
                    
                    # Compute U2 for all valid j2 paths relative to this j1
                    U2_f = u1_f_curr[:, jnp.newaxis, :, :] * psi_j2_vals[jnp.newaxis, :, :, :]
                    U2 = jnp.abs(jnp.fft.ifft2(U2_f))
                    
                    # S2 = U2 * phi
                    U2_f_new = jnp.fft.fft2(U2)
                    s2_f = U2_f_new * self.phi_val[jnp.newaxis, jnp.newaxis, :, :]
                    s2 = jnp.real(jnp.fft.ifft2(s2_f))
                    S2_list.append(s2)
                    
                if len(S2_list) > 0:
                    out['S2'] = jnp.concatenate(S2_list, axis=1)
                else:
                    out['S2'] = jnp.empty((x.shape[0], 0, self.M, self.N))
                    
        return out

    def transform_tileset(self, tileset):
        """
        Compute the 2D scattering transform across all tiles in a generic TileSet.
        
        Parameters
        ----------
        tileset : TileSet
            The input tileset containing apodized patches.
        
        Returns
        -------
        coeffs : dict
            A dictionary containing numpy arrays with the scattering coefficients:
            - 'S0' : array of shape (N_tiles, H, W)
            - 'S1' : array of shape (N_tiles, N_paths_level1, H, W)
            - 'S2' : array of shape (N_tiles, N_paths_level2, H, W)
        """
        import numpy as np
        
        # If polarization, we might have shape (N，2, M, N), otherwise (N, M, N)
        if tileset.pol:
            # Flatten to (N*2, M, N)
            N_t, P, H, W = tileset.data.shape
            x_in = tileset.data.reshape(N_t * P, H, W)
        else:
            x_in = tileset.data
            N_t, H, W = x_in.shape
            P = 1
            
        if H != self.M or W != self.N:
            raise ValueError(f"TileSet dimensions ({H}, {W}) do not match Scattering2D dimensions ({self.M}, {self.N})")
            
        # JAX requires data transfer to device
        x_jax = jnp.array(x_in)
        
        # Execute JIT-compiled transform vectorised over the full batch
        out_jax = self._transform_jit(x_jax)
        
        # Transfer back to host numpy memory
        coeffs = {}
        if tileset.pol:
            coeffs['S0'] = np.array(out_jax['S0']).reshape(N_t, P, H, W)
            if 'S1' in out_jax:
                n_p1 = out_jax['S1'].shape[1]
                coeffs['S1'] = np.array(out_jax['S1']).reshape(N_t, P, n_p1, H, W)
            if 'S2' in out_jax:
                n_p2 = out_jax['S2'].shape[1]
                coeffs['S2'] = np.array(out_jax['S2']).reshape(N_t, P, n_p2, H, W)
        else:
            coeffs['S0'] = np.array(out_jax['S0'])
            if 'S1' in out_jax:
                coeffs['S1'] = np.array(out_jax['S1'])
            if 'S2' in out_jax:
                coeffs['S2'] = np.array(out_jax['S2'])
                
        return coeffs

    def transform_tile(self, tile_data):
        """
        Compute the 2D scattering transform on a single tile array.
        
        Parameters
        ----------
        tile_data : ndarray
            Array of shape (H, W) for intensity, or (P, H, W) for polarisation.
            Must match the (M, N) spatial dimensions of the Scattering2D instance.
            
        Returns
        -------
        coeffs : dict
            A dictionary containing numpy arrays with the scattering coefficients:
            - 'S0' : array of shape (H, W) or (P, H, W)
            - 'S1' : array of shape (N_paths_level1, H, W) or (P, N_paths_level1, H, W)
            - 'S2' : array of shape (N_paths_level2, H, W) or (P, N_paths_level2, H, W)
        """
        import numpy as np
        
        is_pol = (tile_data.ndim == 3)
        if is_pol:
            P, H, W = tile_data.shape
            x_in = tile_data
        else:
            H, W = tile_data.shape
            x_in = tile_data[np.newaxis, :, :]
            
        if H != self.M or W != self.N:
            raise ValueError(f"Tile dimensions ({H}, {W}) do not match Scattering2D dimensions ({self.M}, {self.N})")
            
        x_jax = jnp.array(x_in)
        out_jax = self._transform_jit(x_jax)
        
        coeffs = {}
        for key in out_jax:
            arr = np.array(out_jax[key])
            if not is_pol:
                # Remove the synthetic batch dimension B=1
                coeffs[key] = arr[0]
            else:
                # Shape is naturally (P, ...) which is perfectly aligned
                coeffs[key] = arr
                
        return coeffs
