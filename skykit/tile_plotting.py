import numpy as np
import healpy as hp
from astropy.wcs import WCS
import matplotlib.pyplot as plt

def get_tile_wcs(tileset, face, tx, ty, coord='G'):
    """
    Construct an astropy WCS object for the tangent-plane (gnomonic)
    projection centred on this tile.

    The WCS maps tile pixel coordinates (0-indexed, with the
    convention that pixel centres are at integer coordinates) to
    sky coordinates via a TAN projection.

    The pixel scale and orientation are derived from the actual 3D
    geometry of the HEALPix face-local axes, projected onto the
    tangent plane.  This is robust at all sky positions including
    the poles.

    The reference pixel (CRPIX) is placed at the tile-centre
    pixel, which sits at position (margin + tile_nside//2,
    margin + tile_nside//2) in the tile array (1-indexed for FITS).

    Parameters
    ----------
    face, tx, ty : int
        Tile identifiers.
    coord : str, optional
        Coordinate system.  ``'G'`` (default) for Galactic
        (GLON/GLAT), ``'C'`` for Celestial/Equatorial (RA/Dec
        ICRS).  No other coordinate systems are supported.

    Returns
    -------
    w : astropy.wcs.WCS
        2D celestial WCS with TAN projection.

    Raises
    ------
    ValueError
        If *coord* is not ``'G'`` or ``'C'``.
    """

    if coord not in ('G', 'C'):
        raise ValueError(
            f"Unsupported coord='{coord}'. Use 'G' (Galactic) "
            f"or 'C' (Celestial)."
        )

    # --- Tile centre ---
    ix_c = tx * tileset.tile_nside + tileset.tile_nside // 2
    iy_c = ty * tileset.tile_nside + tileset.tile_nside // 2
    ipix_center = hp.xyf2pix(tileset.nside, ix_c, iy_c, face, nest=True)
    theta_c, phi_c = hp.pix2ang(tileset.nside, ipix_center, nest=True)
    vec_c = np.array(hp.pix2vec(tileset.nside, ipix_center, nest=True))

    lon_c = np.degrees(phi_c)
    lat_c = 90.0 - np.degrees(theta_c)

    # --- Build an orthonormal basis for the tangent plane ---
    #
    # e_lat  = -d(rhat)/d(theta) = local north (toward +lat)
    # e_lon  =  d(rhat)/d(phi) / sin(theta) = local east (toward +lon)
    #
    # At the poles sin(theta) = 0 so we need a fallback.

    sin_theta = np.sin(theta_c)
    cos_theta = np.cos(theta_c)
    sin_phi = np.sin(phi_c)
    cos_phi = np.cos(phi_c)

    if sin_theta > 1e-10:
        # Normal case: well away from the poles
        e_lon = np.array([-sin_phi, cos_phi, 0.0])
        e_lat = np.array([-cos_theta * cos_phi,
                        -cos_theta * sin_phi,
                        sin_theta])
    else:
        # At or very near a pole: pick a consistent basis.
        # At the north pole (theta~0), local east ~ +y, local north ~ -x
        # At the south pole (theta~pi), local east ~ +y, local north ~ +x
        # Use a Gram-Schmidt-like approach from an arbitrary seed.
        if abs(vec_c[2]) < 0.9:
            seed = np.array([0.0, 0.0, 1.0])
        else:
            seed = np.array([1.0, 0.0, 0.0])
        e_lon = np.cross(vec_c, seed)
        e_lon /= np.linalg.norm(e_lon)
        e_lat = np.cross(vec_c, e_lon)
        # Ensure e_lat points toward north (positive z component for
        # north pole, negative for south pole, but more generally
        # toward decreasing theta).
        # The true e_lat = -d(rhat)/d(theta).  At the north pole
        # this should have a negative x or y component... just use
        # the right-hand rule: e_lat = rhat x e_lon already gives
        # the correct orientation by construction since
        # (e_lon, e_lat, rhat) is right-handed like (phi, theta, r)
        # Wait: we want (e_lon, e_lat, rhat) to be right-handed.
        # rhat x e_lon gives a vector perpendicular to both.
        # Check: for standard spherical coords, e_phi x e_theta = -rhat
        # So (e_lon, e_lat, rhat) right-handed means e_lon x e_lat = rhat
        # => e_lat = -(rhat x e_lon)  ... let's just be careful.
        e_lat = -np.cross(vec_c, e_lon)
        e_lat /= np.linalg.norm(e_lat)

    # --- Compute face-local axis directions on the tangent plane ---
    #
    # Get the 3D displacement vector for stepping +1 in ix and iy
    # on the face, then project onto (e_lon, e_lat).

    face_int = int(face)
    ix_c_int = int(ix_c)
    iy_c_int = int(iy_c)

    # ix direction
    ix_nb = min(ix_c_int + 1, tileset.nside - 1)
    if ix_nb == ix_c_int:
        ix_nb = max(ix_c_int - 1, 0)
        sign_x = -1.0
    else:
        sign_x = 1.0

    vec_nb_x = np.array(hp.pix2vec(
        tileset.nside,
        hp.xyf2pix(tileset.nside, ix_nb, iy_c_int, face_int, nest=True),
        nest=True,
    ))
    dvec_x = sign_x * (vec_nb_x - vec_c)

    # iy direction
    iy_nb = min(iy_c_int + 1, tileset.nside - 1)
    if iy_nb == iy_c_int:
        iy_nb = max(iy_c_int - 1, 0)
        sign_y = -1.0
    else:
        sign_y = 1.0

    vec_nb_y = np.array(hp.pix2vec(
        tileset.nside,
        hp.xyf2pix(tileset.nside, ix_c_int, iy_nb, face_int, nest=True),
        nest=True,
    ))
    dvec_y = sign_y * (vec_nb_y - vec_c)

    # Project onto tangent-plane basis
    # dx_lon = dvec . e_lon  (angular displacement in longitude direction)
    # dx_lat = dvec . e_lat  (angular displacement in latitude direction)

    dx_lon_ix = np.dot(dvec_x, e_lon)
    dx_lat_ix = np.dot(dvec_x, e_lat)

    dx_lon_iy = np.dot(dvec_y, e_lon)
    dx_lat_iy = np.dot(dvec_y, e_lat)

    # These tangent-plane displacements are in radians per pixel.
    # Convert to degrees per pixel for the CD matrix.
    dx_lon_ix = np.degrees(dx_lon_ix)
    dx_lat_ix = np.degrees(dx_lat_ix)
    dx_lon_iy = np.degrees(dx_lon_iy)
    dx_lat_iy = np.degrees(dx_lat_iy)

    # --- Build the CD matrix ---
    #
    # The FITS TAN projection maps pixel offsets (di, dj) to
    # tangent-plane offsets (dlon, dlat) via the CD matrix:
    #
    #   dlon = CD1_1 * di + CD1_2 * dj
    #   dlat = CD2_1 * di + CD2_2 * dj
    #
    # where di, dj are in pixel units along NAXIS1 and NAXIS2.
    # In our tile, NAXIS1 corresponds to the ix face direction
    # and NAXIS2 corresponds to the iy face direction.
    #
    # The TAN projection has native longitude increasing to the
    # left, so for a standard WCS the longitude component gets
    # a sign flip.  However, we are computing the actual projected
    # displacement directly, and the sign is already correct if we
    # note that in the tangent plane of a TAN projection, the
    # intermediate world coordinate for longitude is
    #   x = -cos(lat) * sin(dlon) ≈ -dlon  for small dlon
    # So the CD matrix should map pixel offset -> (-dlon, dlat):

    cd = np.array([
        [-dx_lon_ix, -dx_lon_iy],
        [ dx_lat_ix,  dx_lat_iy],
    ])

    # Reference pixel: tile centre in 1-indexed FITS coordinates
    crpix1 = tileset.margin + tileset.tile_nside // 2 + 1.0
    crpix2 = tileset.margin + tileset.tile_nside // 2 + 1.0

    # Coordinate-dependent WCS metadata
    if coord == 'G':
        ctype = ["GLON-TAN", "GLAT-TAN"]
    else:
        ctype = ["RA---TAN", "DEC--TAN"]

    w = WCS(naxis=2)
    w.wcs.crpix = [crpix1, crpix2]
    w.wcs.crval = [lon_c, lat_c]
    w.wcs.cd = cd
    w.wcs.ctype = ctype
    w.wcs.cunit = ["deg", "deg"]

    if coord == 'C':
        w.wcs.radesys = "ICRS"
        w.wcs.equinox = 2000.0

    w.pixel_shape = (tileset.tile_full, tileset.tile_full)

    return w


def plot_tile_flat(tileset, face, tx, ty, dpi=None, title=None, **kwargs):
    '''Plot a specific tile by index as a flat image array without WCS projection, 
    overlaying an outline for the interior of the tile.'''
    tile_data = tileset.get_tile(face, tx, ty)
    
    fig_kwargs = {}
    if dpi is not None:
        fig_kwargs['dpi'] = dpi
        
    base_title = title if title is not None else f'Face {face} tx {tx} ty {ty} (Flat)'
    
    m = tileset.margin
    s = tileset.tile_nside
    box_x = [m - 0.5, m + s - 0.5, m + s - 0.5, m - 0.5, m - 0.5]
    box_y = [m - 0.5, m - 0.5, m + s - 0.5, m + s - 0.5, m - 0.5]

    if tileset.pol:
        fig = plt.figure(figsize=(10, 4), **fig_kwargs)
        ax1 = fig.add_subplot(121)
        im1 = ax1.imshow(tile_data[0].T, origin='lower', **kwargs)
        ax1.plot(box_x, box_y, color='k', alpha=0.5, linestyle='-', linewidth=0.5)
        ax1.set_title(f'{base_title} - Stokes Q')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        ax2 = fig.add_subplot(122)
        im2 = ax2.imshow(tile_data[1].T, origin='lower', **kwargs)
        ax2.plot(box_x, box_y, color='k', alpha=0.5, linestyle='-', linewidth=0.5)
        ax2.set_title(f'{base_title} - Stokes U')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    else:
        fig = plt.figure(**fig_kwargs)
        ax = fig.add_subplot(111)
        im = ax.imshow(tile_data.T, origin='lower', **kwargs)
        ax.plot(box_x, box_y, color='k', alpha=0.5, linestyle='-', linewidth=0.5)
        ax.set_title(base_title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig

def plot_tile_proj(tileset, face, tx, ty, coord='G', dpi=None, title=None, **kwargs):
    '''Plot a specific tile by index using pcolormesh and its WCS for proper spherical wrapping.'''
    wcs = get_tile_wcs(tileset, face, tx, ty, coord=coord)
    tile_data = tileset.get_tile(face, tx, ty)
    
    fig_kwargs = {}
    if dpi is not None:
        fig_kwargs['dpi'] = dpi
        
    base_title = title if title is not None else f'Face {face} tx {tx} ty {ty} (Proj)'
    
    n = tileset.tile_full
    x_edges = np.arange(n + 1) - 0.5
    y_edges = np.arange(n + 1) - 0.5
    X, Y = np.meshgrid(x_edges, y_edges)
    lon_edges, lat_edges = wcs.pixel_to_world_values(X, Y)
    
    if tileset.pol:
        fig = plt.figure(figsize=(10, 4), **fig_kwargs)
        ax1 = fig.add_subplot(121, projection=wcs)
        im1 = ax1.pcolormesh(lon_edges, lat_edges, tile_data[0].T, transform=ax1.get_transform('world'), **kwargs)
        ax1.set_title(f'{base_title} - Stokes Q')
        ax1.grid(color='white', ls='solid', alpha=0.5)
        ax1.set_aspect('equal')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        ax2 = fig.add_subplot(122, projection=wcs)
        im2 = ax2.pcolormesh(lon_edges, lat_edges, tile_data[1].T, transform=ax2.get_transform('world'), **kwargs)
        ax2.set_title(f'{base_title} - Stokes U')
        ax2.grid(color='white', ls='solid', alpha=0.5)
        ax2.set_aspect('equal')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    else:
        fig = plt.figure(**fig_kwargs)
        ax = fig.add_subplot(111, projection=wcs)
        im = ax.pcolormesh(lon_edges, lat_edges, tile_data.T, transform=ax.get_transform('world'), **kwargs)
        ax.set_title(base_title)
        ax.grid(color='white', ls='solid', alpha=0.5)
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig

def plot_tile_flat_at(tileset, lon, lat, dpi=None, title=None, **kwargs):
    '''Plot the tile containing the specified angular coordinates (flat).'''
    face, tx, ty = tileset.tile_containing(lon, lat)
    return plot_tile_flat(tileset, face, tx, ty, dpi=dpi, title=title, **kwargs)

def plot_tile_proj_at(tileset, lon, lat, coord='G', dpi=None, title=None, **kwargs):
    '''Plot the tile containing the specified angular coordinates (projected).'''
    face, tx, ty = tileset.tile_containing(lon, lat)
    return plot_tile_proj(tileset, face, tx, ty, coord=coord, dpi=dpi, title=title, **kwargs)


def plot_scattering_coefs(tileset, coeffs, face, tx, ty, order=1, path_idx=0, coord='G', dpi=None, title=None, **kwargs):
    '''
    Plot a specific scattering coefficient map for a tile using its WCS.

    Parameters
    ----------
    tileset : TileSet
        The TileSet object used to generate the scattering coefficients.
    coeffs : dict
        The output dictionary from Scattering2D containing 'S0', 'S1', etc.
    face, tx, ty : int
        Tile identifiers to plot.
    order : int
        The scattering order to plot (0, 1, or 2).
    path_idx : int
        The index of the scattering path/frequency to plot (for orders 1 and 2).
    coord : str
        Coordinate system ('G' or 'C') for the WCS plot.
    '''
    
    idx = tileset.tile_index(face, tx, ty)
    wcs = get_tile_wcs(tileset, face, tx, ty, coord=coord)
    
    fig_kwargs = {}
    if dpi is not None:
        fig_kwargs['dpi'] = dpi
        
    if order == 0:
        if tileset.pol:
            tile_data = coeffs['S0'][idx] # Shape (P, H, W)
        else:
            tile_data = coeffs['S0'][idx] # Shape (H, W)
        title_suffix = 'S0'
    else:
        key = f'S{order}'
        if key not in coeffs:
            raise KeyError(f'Order {order} coefficients not found in dict.')
        if tileset.pol:
            tile_data = coeffs[key][idx, :, path_idx, :, :] # Shape (P, H, W)
        else:
            tile_data = coeffs[key][idx, path_idx, :, :] # Shape (H, W)
        title_suffix = f'S{order} (path index {path_idx})'
        
    base_title = title if title is not None else f'Face {face} tx {tx} ty {ty}'
        
    if tileset.pol:
        fig = plt.figure(figsize=(10, 4), **fig_kwargs)
        
        ax1 = fig.add_subplot(121, projection=wcs)
        im1 = ax1.imshow(tile_data[0].T, origin='lower', **kwargs)
        ax1.set_title(f'{base_title}\nStokes Q - {title_suffix}')
        ax1.grid(color='white', ls='solid', alpha=0.5)
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        ax2 = fig.add_subplot(122, projection=wcs)
        im2 = ax2.imshow(tile_data[1].T, origin='lower', **kwargs)
        ax2.set_title(f'{base_title}\nStokes U - {title_suffix}')
        ax2.grid(color='white', ls='solid', alpha=0.5)
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    else:
        fig = plt.figure(**fig_kwargs)
        ax = fig.add_subplot(111, projection=wcs)
        im = ax.imshow(tile_data.T, origin='lower', **kwargs)
        ax.set_title(f'{base_title}\n{title_suffix}')
        ax.grid(color='white', ls='solid', alpha=0.5)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    return fig

