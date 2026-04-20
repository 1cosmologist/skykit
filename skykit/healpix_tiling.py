"""
healpix_tiling: Decompose HEALPix maps into overlapping square tiles and back.

Supports scalar (intensity/temperature) maps and spin-2 polarisation
(Stokes Q/U) maps.  For polarisation, Q and U values are parallel-
transported to a common reference frame at the tile centre using
quaternion rotations, so that each tile has a self-consistent
polarisation convention.

Border pixels are resolved by exact topological boundary unfolding. Any gaps in
the margins (topological singularities) are rebinned by averaging the 
ambiguous mappings. This acts directly on real pixel values (no interpolation).
"""

import numpy as np
import healpy as hp

import jax
import jax.numpy as jnp

from .tileset import TileSet

# ---------------------------------------------------------------------------
# Exact Topological unwrapping (JAX)
# ---------------------------------------------------------------------------

@jax.jit
def _resolve_x_jax(f, x, y, nside):
    """
    Resolve HEALPix face boundaries when the x-coordinate overflows or underflows.
    """
    # Initialize the new coordinates and face arrays
    f_new, x_new, y_new = f, x, y
    
    # --- Positive X overflow (x >= nside) ---
    # Create mask for pixels that overflow on the right
    m_xg = x >= nside
    # Sub-masks for the specific 3 horizontal rings of HEALPix faces
    m_xg_f0 = m_xg & (f < 4)               # North polar faces (0, 1, 2, 3)
    m_xg_f4 = m_xg & (f >= 4) & (f < 8)    # Equatorial faces (4, 5, 6, 7)
    m_xg_f8 = m_xg & (f >= 8)              # South polar faces (8, 9, 10, 11)
    
    # For north polar faces, an overflow in +x wraps around the top cap radially.
    # We move to the adjacent face clockwise (f + 1 mod 4).
    # The local x,y coordinates rotate by 90 degrees.
    f_new = jnp.where(m_xg_f0, (f + 1) % 4, f_new)
    y_new = jnp.where(m_xg_f0, 2*nside - 1 - x, y_new)
    x_new = jnp.where(m_xg_f0, y, x_new)
    
    # For equatorial faces, an overflow in +x spills directly up into the north 
    # polar face immediately above it (face index - 4). The x-coord wraps to 0.
    f_new = jnp.where(m_xg_f4, f - 4, f_new)
    x_new = jnp.where(m_xg_f4, x - nside, x_new)
    
    # For south polar faces, an overflow in +x spills upward into the equatorial band.
    # Due to the staggered face arrangement, face 8 goes to 5, 9 to 6, etc.
    map_f8_xg = jnp.array([0,1,2,3,4,5,6,7,5,6,7,4])
    f_new = jnp.where(m_xg_f8, map_f8_xg[f], f_new)
    x_new = jnp.where(m_xg_f8, x - nside, x_new)
    
    # --- Negative X underflow (x < 0) ---
    # Create mask for pixels that underflow on the left
    m_xl = (x < 0) & (~m_xg) 
    # Sub-masks for the specific 3 horizontal rings of HEALPix faces
    m_xl_f0 = m_xl & (f < 4)               # North polar faces
    m_xl_f4 = m_xl & (f >= 4) & (f < 8)    # Equatorial faces
    m_xl_f8 = m_xl & (f >= 8)              # South polar faces
    
    # For north polar faces, an underflow in -x spills straight downward into 
    # the equatorial face immediately below it (face index + 4).
    f_new = jnp.where(m_xl_f0, f + 4, f_new)
    x_new = jnp.where(m_xl_f0, x + nside, x_new)
    
    # For equatorial faces, an underflow in -x spills downward into the south 
    # polar band. Due to face staggering, face 4 goes to 11, 5 to 8, etc.
    map_f4_xl = jnp.array([0,1,2,3,11,8,9,10,0,0,0,0])
    f_new = jnp.where(m_xl_f4, map_f4_xl[f], f_new)
    x_new = jnp.where(m_xl_f4, x + nside, x_new)
    
    # For south polar faces, an underflow in -x wraps around the interior of the 
    # bottom cap. We move to the adjacent face counter-clockwise.
    # The local x,y coordinates rotate by -90 degrees.
    map_f8_xl = jnp.array([0,0,0,0,0,0,0,0,11,8,9,10])
    f_new = jnp.where(m_xl_f8, map_f8_xl[f], f_new)
    y_new = jnp.where(m_xl_f8, -1 - x, y_new)
    x_new = jnp.where(m_xl_f8, y, x_new)

    return f_new, x_new, y_new

@jax.jit
def _resolve_y_jax(f, x, y, nside):
    """
    Resolve HEALPix face boundaries when the y-coordinate overflows or underflows.
    """
    # Initialize the new coordinates and face arrays
    f_new, x_new, y_new = f, x, y
    
    # --- Positive Y overflow (y >= nside) ---
    # Create mask for pixels that overflow upward
    m_yg = y >= nside
    # Sub-masks for the specific 3 horizontal rings of HEALPix faces
    m_yg_f0 = m_yg & (f < 4)               # North polar faces (0, 1, 2, 3)
    m_yg_f4 = m_yg & (f >= 4) & (f < 8)    # Equatorial faces (4, 5, 6, 7)
    m_yg_f8 = m_yg & (f >= 8)              # South polar faces (8, 9, 10, 11)
    
    # For north polar faces, an overflow in +y wraps around the top cap radially.
    # We move to the adjacent face counter-clockwise (f - 1 mod 4).
    # The local x,y coordinates rotate by -90 degrees.
    f_new = jnp.where(m_yg_f0, (f - 1) % 4, f_new)
    x_new = jnp.where(m_yg_f0, 2*nside - 1 - y, x_new)
    y_new = jnp.where(m_yg_f0, x, y_new)
    
    # For equatorial faces, an overflow in +y spills upward into the north 
    # polar band. Due to face staggering, face 4 goes to 3, 5 to 0, etc.
    map_f4_yg = jnp.array([0,1,2,3,3,0,1,2,0,0,0,0])
    f_new = jnp.where(m_yg_f4, map_f4_yg[f], f_new)
    y_new = jnp.where(m_yg_f4, y - nside, y_new)
    
    # For south polar faces, an overflow in +y spills right up into the equatorial 
    # face immediately above it (face index - 4). The y-coord wraps to 0.
    f_new = jnp.where(m_yg_f8, f - 4, f_new)
    y_new = jnp.where(m_yg_f8, y - nside, y_new)
    
    # --- Negative Y underflow (y < 0) ---
    # Create mask for pixels that underflow downward
    m_yl = (y < 0) & (~m_yg)
    # Sub-masks for the specific 3 horizontal rings of HEALPix faces
    m_yl_f0 = m_yl & (f < 4)               # North polar faces
    m_yl_f4 = m_yl & (f >= 4) & (f < 8)    # Equatorial faces
    m_yl_f8 = m_yl & (f >= 8)              # South polar faces
    
    # For north polar faces, an underflow in -y spills completely down into the 
    # equatorial band. Due to the staggering, face 0 goes to 5, 1 to 6, etc.
    map_f0_yl = jnp.array([5,6,7,4,0,0,0,0,0,0,0,0])
    f_new = jnp.where(m_yl_f0, map_f0_yl[f], f_new)
    y_new = jnp.where(m_yl_f0, y + nside, y_new)
    
    # For equatorial faces, an underflow in -y spills straight downward into 
    # the south polar face immediately below it (face index + 4).
    f_new = jnp.where(m_yl_f4, f + 4, f_new)
    y_new = jnp.where(m_yl_f4, y + nside, y_new)
    
    # For south polar faces, an underflow in -y wraps around the interior of the 
    # bottom cap. We move to the adjacent face clockwise.
    # The local x,y coordinates rotate by +90 degrees.
    map_f8_yl = jnp.array([0,0,0,0,0,0,0,0,9,10,11,8])
    f_new = jnp.where(m_yl_f8, map_f8_yl[f], f_new)
    x_new = jnp.where(m_yl_f8, -1 - y, x_new)
    y_new = jnp.where(m_yl_f8, x, y_new)
    
    return f_new, x_new, y_new

@jax.jit
def _unwrap_step_xy_jax(carry):
    f, x, y, nside = carry
    f, x, y = _resolve_x_jax(f, x, y, nside)
    f, x, y = _resolve_y_jax(f, x, y, nside)
    return (f, x, y, nside)

@jax.jit
def _unwrap_step_yx_jax(carry):
    f, x, y, nside = carry
    f, x, y = _resolve_y_jax(f, x, y, nside)
    f, x, y = _resolve_x_jax(f, x, y, nside)
    return (f, x, y, nside)

@jax.jit
def _cond_fun_jax(carry):
    f, x, y, nside = carry
    return jnp.any((x < 0) | (x >= nside) | (y < 0) | (y >= nside))

@jax.jit
def _unwrap_grid_vec_xy_jax(f_base, x_arr, y_arr, nside):
    f_arr = jnp.full(x_arr.shape, f_base, dtype=jnp.int32)
    f_res, x_res, y_res, _ = jax.lax.while_loop(_cond_fun_jax, _unwrap_step_xy_jax, (f_arr, x_arr, y_arr, nside))
    return f_res, x_res, y_res

@jax.jit
def _unwrap_grid_vec_yx_jax(f_base, x_arr, y_arr, nside):
    f_arr = jnp.full(x_arr.shape, f_base, dtype=jnp.int32)
    f_res, x_res, y_res, _ = jax.lax.while_loop(_cond_fun_jax, _unwrap_step_yx_jax, (f_arr, x_arr, y_arr, nside))
    return f_res, x_res, y_res

def _unwrap_grid_vec(f_base, IX, IY, nside, order='xy'):
    """
    JAX vectorized exact topological unwrapping.
    Walks out-of-bounds coordinates across HEALPix face boundaries iteratively.
    """
    x_arr = jnp.asarray(IX, dtype=jnp.int32)
    y_arr = jnp.asarray(IY, dtype=jnp.int32)
    
    if order == 'xy':
        f_res, x_res, y_res = _unwrap_grid_vec_xy_jax(f_base, x_arr, y_arr, nside)
    else:
        f_res, x_res, y_res = _unwrap_grid_vec_yx_jax(f_base, x_arr, y_arr, nside)
        
    return np.asarray(f_res, dtype=np.int64), np.asarray(x_res, dtype=np.int64), np.asarray(y_res, dtype=np.int64)


# ---------------------------------------------------------------------------
# Internal: polarisation rotation via quaternions
# ---------------------------------------------------------------------------

def _local_north(theta, phi):
    """
    Unit vector pointing toward the north pole in the tangent plane
    at position (theta, phi) on the unit sphere.

    This is -d(rhat)/d(theta), i.e. the theta-hat direction pointing
    toward decreasing theta (toward the pole).
    """
    nx = -np.cos(theta) * np.cos(phi)
    ny = -np.cos(theta) * np.sin(phi)
    nz = np.sin(theta)
    return nx, ny, nz


def _compute_psi(nside, face, tx, ty, tile_nside, ipix_tile):
    """
    Compute the parallel-transport rotation angle psi at every pixel
    in a tile (fully vectorised).

    psi is the angle from the pixel's HEALPix local north to the
    parallel-transported tile-centre north, measured CCW in the
    tangent plane.  For a spin-2 field Q + iU, a CCW frame rotation
    by psi transforms as Q' + iU' = (Q + iU) * exp(-2i*psi), giving:

        Q' =  Q cos(2 psi) + U sin(2 psi)
        U' = -Q sin(2 psi) + U cos(2 psi)

    transforms from the HEALPix pixel frame to the tile-centre frame.

    The parallel transport is performed by rotating the centre's
    north vector to each pixel along the great circle connecting
    them.  The rotation is defined by:

        axis  = normalize(vec_centre x vec_pixel)
        angle = arccos(vec_centre · vec_pixel)

    and is applied via the Rodrigues rotation formula, which is
    equivalent to quaternion rotation but fully vectorisable.

    Parameters
    ----------
    nside : int
    face, tx, ty : int
    tile_nside : int
    ipix_tile : ndarray of int64 — NESTED pixel index at each tile position.

    Returns
    -------
    psi : ndarray, same shape as ipix_tile — rotation angles in radians.
    """
    orig_shape = ipix_tile.shape
    ipix_flat = ipix_tile.ravel()
    n_pix = len(ipix_flat)

    # --- Tile centre ---
    ix_c = tx * tile_nside + tile_nside // 2
    iy_c = ty * tile_nside + tile_nside // 2
    ipix_center = hp.xyf2pix(nside, ix_c, iy_c, face, nest=True)
    theta_c, phi_c = hp.pix2ang(nside, ipix_center, nest=True)
    vec_center = np.array(hp.pix2vec(nside, ipix_center, nest=True))  # (3,)

    # Local north at centre: shape (3,)
    north_c = np.array(_local_north(theta_c, phi_c))

    # --- All pixel positions: shape (n_pix, 3) ---
    theta_p, phi_p = hp.pix2ang(nside, ipix_flat, nest=True)
    vec_pixels = np.column_stack(hp.pix2vec(nside, ipix_flat, nest=True))

    # Local north at each pixel: shape (n_pix, 3)
    north_p = np.column_stack(_local_north(theta_p, phi_p))

    # --- Great-circle rotation axis and angle ---
    # axis = vec_center × vec_pixel  (n_pix, 3)
    axis = np.cross(vec_center, vec_pixels)
    axis_norm = np.linalg.norm(axis, axis=1, keepdims=True)  # (n_pix, 1)

    # Mask for degenerate cases (coincident or antipodal pixels)
    valid = (axis_norm.ravel() > 1e-15)

    # Safe normalisation (avoid division by zero; degenerate entries
    # will be overwritten with psi=0 later)
    safe_norm = np.where(axis_norm > 1e-15, axis_norm, 1.0)
    axis_hat = axis / safe_norm  # (n_pix, 3)

    # Rotation angle
    dot_cv = np.sum(vec_center * vec_pixels, axis=1)  # (n_pix,)
    angle = np.arccos(np.clip(dot_cv, -1.0, 1.0))     # (n_pix,)

    # --- Rodrigues rotation formula ---
    # v_rot = v cos(a) + (k × v) sin(a) + k (k · v)(1 - cos(a))
    # where v = north_c (broadcast), k = axis_hat, a = angle
    cos_a = np.cos(angle)[:, np.newaxis]   # (n_pix, 1)
    sin_a = np.sin(angle)[:, np.newaxis]   # (n_pix, 1)

    north_c_broad = north_c[np.newaxis, :]  # (1, 3)

    k_cross_v = np.cross(axis_hat, north_c_broad)  # (n_pix, 3)
    k_dot_v = np.sum(axis_hat * north_c_broad, axis=1, keepdims=True)  # (n_pix, 1)

    north_transported = (north_c_broad * cos_a
                         + k_cross_v * sin_a
                         + axis_hat * k_dot_v * (1.0 - cos_a))  # (n_pix, 3)

    # --- Project onto tangent plane (remove radial component) ---
    radial_comp = np.sum(north_transported * vec_pixels, axis=1,
                         keepdims=True)  # (n_pix, 1)
    north_transported -= radial_comp * vec_pixels

    nt_norm = np.linalg.norm(north_transported, axis=1, keepdims=True)
    safe_nt_norm = np.where(nt_norm > 1e-15, nt_norm, 1.0)
    north_transported /= safe_nt_norm

    # --- psi = angle from local north to transported north (CCW) ---
    cos_psi = np.clip(
        np.sum(north_p * north_transported, axis=1), -1.0, 1.0)
    cross_nt = np.cross(north_p, north_transported)  # (n_pix, 3)
    sin_psi = np.sum(cross_nt * vec_pixels, axis=1)
    psi = np.arctan2(sin_psi, cos_psi)

    # --- Zero out degenerate pixels and the centre pixel ---
    psi[~valid] = 0.0
    psi[ipix_flat == ipix_center] = 0.0

    return psi.reshape(orig_shape)


def _apply_spin2_rotation(Q, U, psi):
    """
    Rotate Q, U by frame rotation angle psi (spin-2 convention).

    For a CCW frame rotation by psi, a spin-2 field transforms as
    Q' + iU' = (Q + iU) * exp(-2i*psi):

        Q' =  Q cos(2 psi) + U sin(2 psi)
        U' = -Q sin(2 psi) + U cos(2 psi)
    """
    c2 = np.cos(2.0 * psi)
    s2 = np.sin(2.0 * psi)
    Q_rot = Q * c2 + U * s2
    U_rot = -Q * s2 + U * c2
    return Q_rot, U_rot



# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_inputs(nside, tile_nside, margin):
    """Common validation for tiling parameters with exact margin unwrapping."""
    npix = 12 * nside * nside
    if not hp.isnpixok(npix):
        raise ValueError(f"Invalid nside={nside}. Must be a power of 2.")
    if nside % tile_nside != 0:
        raise ValueError(f"nside ({nside}) must be an exact multiple of tile_nside ({tile_nside}).")
    if margin >= nside:
        raise ValueError(f"margin ({margin}) must be strictly less than nside ({nside}) to safely project.")
    if not (nside > 0 and (nside & (nside - 1)) == 0):
        raise ValueError(f"nside ({nside}) must be a power of 2.")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class HealpixTileProjector:
    """
    Pre-computes and caches pixel mappings and polarization rotation angles
    for HEALPix map and TileSet transformations.

    Use this class to transform multiple maps with the same geometry
    to avoid recalculating coordinates and rotation angles.
    """
    def __init__(self, nside, tile_nside, margin, pol=False):
        _validate_inputs(nside, tile_nside, margin)
        self.nside = nside
        self.tile_nside = tile_nside
        self.margin = margin
        self.pol = pol
        
        self.n_subtiles = nside // tile_nside
        self.tile_full = tile_nside + 2 * margin
        self.n_tiles = 12 * self.n_subtiles * self.n_subtiles
        
        # Pre-allocate arrays
        shape = (self.n_tiles, self.tile_full, self.tile_full)
        self.ipix_xy = np.empty(shape, dtype=np.int64)
        self.ipix_yx = np.empty(shape, dtype=np.int64)
        
        if self.pol:
            self.psi_xy = np.empty(shape, dtype=np.float64)
            self.psi_yx = np.empty(shape, dtype=np.float64)
            
        interior_shape = (self.n_tiles, self.tile_nside, self.tile_nside)
        self.ipix_interior = np.empty(interior_shape, dtype=np.int64)
        if self.pol:
            self.psi_interior = np.empty(interior_shape, dtype=np.float64)

        i_arr = np.arange(self.tile_full)
        II, JJ = np.meshgrid(i_arr, i_arr, indexing="xy")

        idx = 0
        for face in range(12):
            for tx in range(self.n_subtiles):
                for ty in range(self.n_subtiles):
                    x0 = tx * tile_nside
                    y0 = ty * tile_nside
                    
                    # Full tile mapping
                    IX = x0 + II - margin
                    IY = y0 + JJ - margin

                    f_xy, x_xy, y_xy = _unwrap_grid_vec(face, IX, IY, nside, 'xy')
                    f_yx, x_yx, y_yx = _unwrap_grid_vec(face, IX, IY, nside, 'yx')

                    ipix_xy = hp.xyf2pix(nside, x_xy, y_xy, f_xy, nest=True)
                    ipix_yx = hp.xyf2pix(nside, x_yx, y_yx, f_yx, nest=True)

                    self.ipix_xy[idx] = ipix_xy
                    self.ipix_yx[idx] = ipix_yx

                    if self.pol:
                        psi_xy = _compute_psi(nside, face, tx, ty, tile_nside, ipix_xy)
                        psi_yx = _compute_psi(nside, face, tx, ty, tile_nside, ipix_yx)
                        self.psi_xy[idx] = psi_xy
                        self.psi_yx[idx] = psi_yx
                        
                    # Interior tile mapping (for tiles2map)
                    m = margin
                    s = tile_nside
                    ix_arr = np.arange(tx * tile_nside, (tx + 1) * tile_nside, dtype=np.int64)
                    iy_arr = np.arange(ty * tile_nside, (ty + 1) * tile_nside, dtype=np.int64)
                    IX_int, IY_int = np.meshgrid(ix_arr, iy_arr, indexing="xy")

                    face_arr = np.full(IX_int.size, face, dtype=np.int64)
                    ipix_int = hp.xyf2pix(nside, IX_int.ravel(), IY_int.ravel(), face_arr, nest=True)
                    self.ipix_interior[idx] = ipix_int.reshape((s, s))
                    
                    if self.pol:
                        self.psi_interior[idx] = self.psi_xy[idx, m: m + s, m: m + s]

                    idx += 1

    def map2tiles(self, healpix_map, nested=False):
        """
        Decompose a HEALPix map into overlapping square tiles.
        """
        npix = 12 * self.nside * self.nside
        healpix_map = np.asarray(healpix_map, dtype=np.float64)

        if self.pol:
            if healpix_map.shape != (2, npix):
                raise ValueError(f"For pol=True, map must have shape (2, {npix})")
            if nested:
                nested_q = healpix_map[0]
                nested_u = healpix_map[1]
            else:
                nested_q = hp.reorder(healpix_map[0], r2n=True)
                nested_u = hp.reorder(healpix_map[1], r2n=True)
                
            data = np.empty((self.n_tiles, 2, self.tile_full, self.tile_full), dtype=np.float64)
            
            # Vectorised assignment over precomputed indices
            q_xy, u_xy = _apply_spin2_rotation(nested_q[self.ipix_xy], nested_u[self.ipix_xy], self.psi_xy)
            q_yx, u_yx = _apply_spin2_rotation(nested_q[self.ipix_yx], nested_u[self.ipix_yx], self.psi_yx)
            
            data[:, 0] = 0.5 * (q_xy + q_yx)
            data[:, 1] = 0.5 * (u_xy + u_yx)
            
            tileset = TileSet(data, self.nside, self.tile_nside, self.margin, pol=self.pol)
            return tileset

        else:
            if healpix_map.shape != (npix,):
                raise ValueError(f"For pol=False, map must have shape ({npix},)")
            if nested:
                nested_map = healpix_map
            else:
                nested_map = hp.reorder(healpix_map, r2n=True)
                
            val_xy = nested_map[self.ipix_xy]
            val_yx = nested_map[self.ipix_yx]
            data = 0.5 * (val_xy + val_yx)
            
            return TileSet(data, self.nside, self.tile_nside, self.margin, pol=self.pol)

    def tiles2map(self, tileset, nested=False):
        """
        Reconstruct a HEALPix map from a TileSet.
        """
        npix = 12 * self.nside * self.nside
        m = self.margin
        s = self.tile_nside

        if self.pol:
            nested_q = np.zeros(npix, dtype=tileset.data.dtype)
            nested_u = np.zeros(npix, dtype=tileset.data.dtype)
            
            q_interior = tileset.data[:, 0, m: m + s, m: m + s]
            u_interior = tileset.data[:, 1, m: m + s, m: m + s]
            
            q_orig, u_orig = _apply_spin2_rotation(q_interior, u_interior, -self.psi_interior)
            
            # Strict guarantee that tile interiors are exact HEALPix partitions (mutually exclusive pixels)
            nested_q[self.ipix_interior.ravel()] = q_orig.ravel()
            nested_u[self.ipix_interior.ravel()] = u_orig.ravel()

            if nested:
                return np.stack([nested_q, nested_u], axis=0)
            else:
                return np.stack([hp.reorder(nested_q, n2r=True), hp.reorder(nested_u, n2r=True)], axis=0)

        else:
            nested_map = np.zeros(npix, dtype=tileset.data.dtype)
            interior = tileset.data[:, m: m + s, m: m + s]
            
            nested_map[self.ipix_interior.ravel()] = interior.ravel()

            if nested:
                return nested_map
            else:
                return hp.reorder(nested_map, n2r=True)


def healpix2tiles(healpix_map, nside, tile_nside, margin, pol=False, nested=False):
    """
    Decompose a HEALPix map into overlapping square tiles, resolving
    margin pixels via exact topological boundary unfolding. Any gaps in
    the margins (topological singularities) are rebinned by averaging the 
    ambiguous mappings. This acts directly on real pixel values (no interpolation).

    Parameters
    ----------
    healpix_map : ndarray
        If ``pol=False``: 1D array of length ``12 * nside**2``.
        If ``pol=True``: shape ``(2, 12 * nside**2)`` where
        ``[0]`` is Stokes Q and ``[1]`` is Stokes U.
    nside : int
        HEALPix resolution parameter (power of 2).
    tile_nside : int
        Interior side length of each tile. Must divide nside.
    margin : int
        Overlap border width. Must be < nside.
    nested : bool
        True if input is NESTED ordering.
    pol : bool
        If True, treat input as a spin-2 (Q, U) field and
        parallel-transport to each tile centre's reference frame.

    Returns
    -------
    tileset : TileSet
    """
    projector = HealpixTileProjector(nside, tile_nside, margin, pol=pol)
    return projector.map2tiles(healpix_map, nested=nested)


def tiles2healpix(tileset, nested=False):
    """
    Reconstruct a HEALPix map from a TileSet.

    Only interior pixels are used; overlap borders are discarded.
    For scalar maps, the round-trip is lossless.
    For polarisation maps, the inverse spin-2 rotation is applied
    to transform Q and U back from the tile-centre frame to the
    HEALPix pixel frame.

    Parameters
    ----------
    tileset : TileSet
    nested : bool
        If True, output is NESTED ordering.

    Returns
    -------
    healpix_map : ndarray
        If ``tileset.pol is False``: 1D array of length ``12*nside**2``.
        If ``tileset.pol is True``: shape ``(2, 12*nside**2)``.
    """
    projector = HealpixTileProjector(tileset.nside, tileset.tile_nside, tileset.margin, pol=tileset.pol)
    return projector.tiles2map(tileset, nested=nested)