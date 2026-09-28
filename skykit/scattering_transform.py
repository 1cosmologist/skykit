"""Undecimated two-dimensional scattering on scalar tile arrays."""

import jax
import jax.numpy as jnp
import numpy as np

class Scattering2D:
    """
    A 2D Wavelet Scattering Transform, defined using JAX for fast execution on GPUs.
    Computes scattering coefficients directly on pre-tiled batches (from a TileSet).
    Uses undecimated formulation (no spatial downsampling) to preserve 1:1 pixel 
    correspondence with the original tiles, allowing easy reconstruction or analysis.

    The filter bank (dict with 'psi' and 'phi' keys) must be generated externally via
    ``generate_filter_bank`` and passed in at construction time. This allows the caller
    to choose and inspect the filter bank before committing to a transform.
    """
    
    def __init__(self, filter_bank, max_order=2):
        """
        Initialize the Scattering Transform from a pre-built filter bank.
        
        Parameters
        ----------
        filter_bank : dict
            Output of ``generate_filter_bank``. Must contain:
            - ``'psi'``: list of dicts, each with keys ``'j'``, ``'theta'``, ``'val'``
              where ``'val'`` is a 2D JAX/numpy array of shape ``(M, N)``.
            - ``'phi'``: dict with key ``'val'``, a 2D array of shape ``(M, N)``.
        max_order : int
            Maximum scattering order to compute (1 or 2).
        """
        self.max_order = max_order

        # Infer spatial dimensions from the first psi filter
        psi0_val = filter_bank['psi'][0]['val']
        self.M, self.N = psi0_val.shape

        # Infer J and L from the psi list
        self.J = int(max(f['j'] for f in filter_bank['psi'])) + 1
        self.L = sum(1 for f in filter_bank['psi'] if f['j'] == 0)

        # Stack all psi filters into a single tensor of shape (N_psi, M, N)
        self.psi_vals = jnp.stack([f['val'] for f in filter_bank['psi']])
        # Ensure primitive Python ints so jax.jit doesn't trace the mask indices
        self.psi_j = np.array([int(f['j']) for f in filter_bank['psi']])

        # Lowpass filter phi shape (M, N)
        self.phi_val = filter_bank['phi']['val']

        # JIT compile the batched transform function
        self._transform_jit = jax.jit(
            self._compute_coefficients,
            static_argnames=('spatial_average', 'return_feature_maps'),
        )

    def _compute_coefficients(self, x, spatial_average=False, return_feature_maps=False):
        """
        Internal JAX function to compute scattering paths on a batch.
        
        Parameters
        ----------
        x : array-like (B, M, N)
            Batch of input patches.
            
        Returns
        -------
        out : dict
            'S0': shape (B, M, N), or (B,) when spatial_average=True
            'S1': shape (B, N_psi, M, N), or (B, N_psi)
            'S2': shape (B, N_paths_order2, M, N), or (B, N_paths_order2)
            Optional 'U1' and 'U2' are unsmoothed feature maps.
        """
        # FFT of the input: (B, M, N)
        x_f = jnp.fft.fft2(x)
        
        # The mean of a periodic convolution depends only on its DC response.
        phi_dc = jnp.real(self.phi_val[0, 0])
        if spatial_average:
            S0 = jnp.mean(x, axis=(-2, -1)) * phi_dc
        else:
            S0_f = x_f * self.phi_val[jnp.newaxis, :, :]
            S0 = jnp.real(jnp.fft.ifft2(S0_f))

        out = {'S0': S0}
        
        if self.max_order >= 1:
            # Order 1: |x * psi1|
            # x_f (B, 1, M, N) * psi_vals (1, N_psi, M, N) -> (B, N_psi, M, N)
            U1_f = x_f[:, jnp.newaxis, :, :] * self.psi_vals[jnp.newaxis, :, :, :]
            U1 = jnp.abs(jnp.fft.ifft2(U1_f))
            if return_feature_maps:
                out['U1'] = U1
            
            U1_f_new = jnp.fft.fft2(U1)
            if spatial_average:
                S1 = jnp.mean(U1, axis=(-2, -1)) * phi_dc
            else:
                S1_f = U1_f_new * self.phi_val[jnp.newaxis, jnp.newaxis, :, :]
                S1 = jnp.real(jnp.fft.ifft2(S1_f))
            out['S1'] = S1
            
            if self.max_order >= 2:
                # Order 2: ||x * psi1| * psi2|
                # We only compute paths where j2 > j1 (frequency decreasing path to avoid energy explosion)
                S2_list = []
                U2_list = []
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
                    if return_feature_maps:
                        U2_list.append(U2)
                    
                    if spatial_average:
                        s2 = jnp.mean(U2, axis=(-2, -1)) * phi_dc
                    else:
                        s2_f = jnp.fft.fft2(U2) * self.phi_val[jnp.newaxis, jnp.newaxis, :, :]
                        s2 = jnp.real(jnp.fft.ifft2(s2_f))
                    S2_list.append(s2)
                    
                if len(S2_list) > 0:
                    out['S2'] = jnp.concatenate(S2_list, axis=1)
                    if return_feature_maps:
                        out['U2'] = jnp.concatenate(U2_list, axis=1)
                else:
                    shape = (x.shape[0], 0) if spatial_average else (x.shape[0], 0, self.M, self.N)
                    out['S2'] = jnp.empty(shape)
                    if return_feature_maps:
                        out['U2'] = jnp.empty((x.shape[0], 0, self.M, self.N))
                    
        return out

    def transform_tileset(self, tileset, *, spatial_average=False, return_feature_maps=False,
                          batch_size=None, feature_map_file=None):
        """
        Compute the 2D scattering transform across all tiles in a generic TileSet.
        
        Parameters
        ----------
        tileset : TileSet
            The input tileset containing apodized patches.
        spatial_average : bool, optional
            Return one mean value per tile and scattering path instead of a
            coefficient map. The mean includes the full tile, including margins.
        return_feature_maps : bool, optional
            Also return unsmoothed modulus maps as 'U1' and 'U2'. These keep
            their spatial dimensions even when spatial_average=True.
        batch_size : int, optional
            Number of tiles to transform at once. Useful for large tile sets.
        feature_map_file : path-like, optional
            Write U1 and U2 directly to one HDF5 file, in tile-sized chunks.
            Requires ``return_feature_maps=True``. The returned dictionary
            contains the file path under ``'feature_map_file'`` in place of U
            arrays; pass that path to ``compute_scattering_statistics``.
        
        Returns
        -------
        coeffs : dict
            A dictionary containing numpy arrays with the scattering coefficients:
            - 'S0' : shape (N_tiles, H, W), or (N_tiles,) if spatially averaged
            - 'S1' : shape (N_tiles, N_paths_level1, H, W), or (N_tiles, N_paths_level1)
            - 'S2' : shape (N_tiles, N_paths_level2, H, W), or (N_tiles, N_paths_level2)
            Polarisation adds a size-2 axis after N_tiles. Optional 'U1' and
            'U2' have the corresponding unaveraged map shapes when no output
            file is requested.
        """
        from contextlib import nullcontext
        from pathlib import Path
        import h5py
        import numpy as np
        from .feature_statistics import _create_dataset

        if feature_map_file is not None and not return_feature_maps:
            raise ValueError("feature_map_file requires return_feature_maps=True")
        if batch_size is not None and (not isinstance(batch_size, int) or batch_size < 1):
            raise ValueError("batch_size must be a positive integer")
        
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
            
        coeffs = {}
        chunk_size = batch_size or (1 if feature_map_file is not None else N_t)
        filepath = Path(feature_map_file) if feature_map_file is not None else None
        context = h5py.File(filepath, "w") if filepath is not None else nullcontext(None)
        with context as handle:
            if handle is not None:
                handle.attrs["single_tile"] = False
                handle.attrs["pol"] = bool(tileset.pol)
                handle.attrs["nside"] = tileset.nside
                handle.attrs["tile_nside"] = tileset.tile_nside
                handle.attrs["margin"] = tileset.margin
            for start in range(0, N_t, chunk_size):
                stop = min(start + chunk_size, N_t)
                batch = x_in[start * P:stop * P] if tileset.pol else x_in[start:stop]
                out_jax = self._transform_jit(
                    jnp.asarray(batch), spatial_average=spatial_average,
                    return_feature_maps=return_feature_maps,
                )
                for key, value in out_jax.items():
                    arr = np.asarray(value)
                    if tileset.pol:
                        arr = arr.reshape((stop - start, P) + arr.shape[1:])
                    if handle is not None and key in ("U1", "U2"):
                        if key not in handle:
                            _create_dataset(handle, key, (N_t,) + arr.shape[1:], arr.dtype,
                                            single_tile=False, polarized=tileset.pol)
                        handle[key][start:stop] = arr
                    else:
                        if key not in coeffs:
                            coeffs[key] = np.empty((N_t,) + arr.shape[1:], dtype=arr.dtype)
                        coeffs[key][start:stop] = arr
        if filepath is not None:
            coeffs["feature_map_file"] = filepath
                
        return coeffs

    def transform_tile(self, tile_data, *, spatial_average=False, return_feature_maps=False,
                       feature_map_file=None):
        """
        Compute the 2D scattering transform on a single tile array.
        
        Parameters
        ----------
        tile_data : ndarray
            Array of shape (H, W) for intensity, or (P, H, W) for polarisation.
            Must match the (M, N) spatial dimensions of the Scattering2D instance.
        spatial_average : bool, optional
            Return one mean value per scattering path over the full tile.
        return_feature_maps : bool, optional
            Also return unsmoothed modulus maps as 'U1' and 'U2'. These keep
            their spatial dimensions even when spatial_average=True.
        feature_map_file : path-like, optional
            Save U1 and U2 for this tile to one HDF5 file. Requires
            ``return_feature_maps=True``. The returned dictionary includes
            ``'feature_map_file'`` instead of in-memory U arrays.
            
        Returns
        -------
        coeffs : dict
            Contains ``S0``, ``S1``, and ``S2`` as NumPy arrays.
            ``S0`` has shape ``(H, W)`` or ``(P, H, W)``;
            ``S1`` has a scattering-path axis before the spatial axes,
            and ``S2`` is analogous. With ``spatial_average=True`` the
            spatial axes are removed. Optional ``U1`` and ``U2`` always
            retain their spatial axes.
        """
        import numpy as np
        from pathlib import Path
        from .feature_statistics import save_feature_maps

        if feature_map_file is not None and not return_feature_maps:
            raise ValueError("feature_map_file requires return_feature_maps=True")
        
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
        out_jax = self._transform_jit(
            x_jax, spatial_average=spatial_average,
            return_feature_maps=return_feature_maps,
        )
        
        coeffs = {}
        for key in out_jax:
            arr = np.array(out_jax[key])
            if not is_pol:
                # Remove the synthetic batch dimension B=1
                coeffs[key] = arr[0]
            else:
                # Shape is naturally (P, ...) which is perfectly aligned
                coeffs[key] = arr
        if feature_map_file is not None:
            filepath = save_feature_maps(coeffs, feature_map_file, single_tile=True)
            coeffs.pop("U1", None)
            coeffs.pop("U2", None)
            coeffs["feature_map_file"] = Path(filepath)
                
        return coeffs
