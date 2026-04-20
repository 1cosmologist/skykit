import numpy as np
import h5py

def create_apodization_window(tile_nside, margin, taper_width=None, taper_type='cosine'):
    """
    Create a 2D apodization (tapering) window for a square tile with margins.
    
    The window is exactly 1.0 in the interior (`tile_nside` x `tile_nside`) 
    and smoothly tapers to 0.0 at the outer boundaries of the `margin`. 
    This enforces periodic boundary conditions on the patch for 2D FFTs, 
    minimising ringing or edge-effect artifacts.

    Parameters
    ----------
    tile_nside : int
        The side length of the interior (untapered) tile.
    margin : int
        The width of the border region where the taper is applied.
    taper_width : int, optional
        The physical length in pixels over which the taper transitions from 0 to 1.
        By default, it is equal to `margin`. Cannot exceed `margin`. 
        If smaller than `margin`, the extra outer pixels are padded with exactly 0.0.
    taper_type : str
        The function used to taper the margins. Options:
        - 'nuttall': A Nuttall window with continuous first derivatives and very low side-lobes.
          Excellent for strongly suppressing spectral leakage in FFTs.
        - 'cosine' (default): A Hann/Tukey-like squared-sine taper. Very smooth 
          derivatives, excellent for isolating the interior.
        - 'sine': A steeper sine quarter-wave taper.

    Returns
    -------
    window_2d : ndarray
        A 2D array of shape (tile_nside + 2*margin, tile_nside + 2*margin) 
        containing the apodization weights [0.0, 1.0].
    """
    if margin == 0:
        return np.ones((tile_nside, tile_nside), dtype=np.float64)

    if taper_width is None:
        taper_width = margin
    if taper_width > margin:
        raise ValueError(f"taper_width ({taper_width}) cannot exceed margin ({margin}).")
    if taper_width < 0:
        raise ValueError("taper_width cannot be negative.")

    # Taper transition region
    x = np.arange(taper_width, dtype=np.float64)

    # Define the 1D taper-up profile based on the chosen type
    if taper_width == 0:
        taper_up = np.array([], dtype=np.float64)
    elif taper_type == 'cosine':
        # Smoothly goes from 0 to 1. Derivative is 0 at both ends.
        taper_up = 0.5 * (1.0 - np.cos(np.pi * x / taper_width))
    elif taper_type == 'sine':
        # Steeper rise, derivative is non-zero at the start but 0 at the top
        taper_up = np.sin((np.pi / 2.0) * (x / taper_width))
    elif taper_type == 'nuttall':
        # Nuttall window (left half) - very low side-lobes
        a0, a1, a2, a3 = 0.355768, 0.487396, 0.144232, 0.012604
        arg = np.pi * x / taper_width
        taper_up = a0 - a1 * np.cos(arg) + a2 * np.cos(2 * arg) - a3 * np.cos(3 * arg)
    else:
        raise ValueError(f"Unknown taper_type '{taper_type}'. Use 'nuttall', 'cosine', or 'sine'.")

    # Pad with zeros beyond the taper width to fill the full margin
    zeros_outer = np.zeros(margin - taper_width, dtype=np.float64)
    left_margin = np.concatenate([zeros_outer, taper_up])

    # The flat interior
    center = np.ones(tile_nside, dtype=np.float64)

    # Taper down is the mirror of taper up
    right_margin = left_margin[::-1]

    # Combine into a single 1D window (length = tile_nside + 2*margin)
    w_1d = np.concatenate([left_margin, center, right_margin])

    # For a square patch, the optimal 2D window is the outer product of the 1D window
    # This ensures tapering is applied independently in X and Y, and naturally 
    # blends the corners down to 0.0.
    window_2d = np.outer(w_1d, w_1d)

    return window_2d


def apply_apodization(tileset, taper_width=None, taper_type='cosine', inplace=False):
    """
    Generate and apply an apodization window to all tiles in a TileSet.

    Parameters
    ----------
    tileset : TileSet
        The TileSet object containing the fields to be tapered.
    taper_width : int, optional
        The width in pixels of the taper. Defaults to `tileset.margin`.
    taper_type : str, optional
        The type of taper to apply ('nuttall', 'cosine', 'sine').
    inplace : bool, optional
        If True, modifies the data array of the input tileset directly to save memory. 
        If False, returns a new TileSet with a copy of the data.

    Returns
    -------
    out_tileset : TileSet
        The TileSet with the apodization window applied to all tiles.
    """
    window = create_apodization_window(tileset.tile_nside, tileset.margin, 
                                       taper_width=taper_width, taper_type=taper_type)
    
    if inplace:
        if tileset.pol:
            tileset.data *= window[np.newaxis, np.newaxis, :, :]
        else:
            tileset.data *= window[np.newaxis, :, :]
        return tileset
    else:
        from .tileset import TileSet
        new_data = tileset.data.copy()
        
        if tileset.pol:
            new_data *= window[np.newaxis, np.newaxis, :, :]
        else:
            new_data *= window[np.newaxis, :, :]
            
        new_tileset = TileSet(new_data, tileset.nside, tileset.tile_nside, 
                              tileset.margin, pol=tileset.pol)
            
        return new_tileset

def write_tileset_hdf5(tileset, filepath):
    """
    Write a TileSet and its key metadata to an HDF5 file.
    
    Parameters
    ----------
    tileset : TileSet
        The TileSet object to serialize.
    filepath : str
        Path of the destination HDF5 file.
    """
    with h5py.File(filepath, 'w') as f:
        # Store metadata attributes
        f.attrs['nside'] = tileset.nside
        f.attrs['tile_nside'] = tileset.tile_nside
        f.attrs['margin'] = tileset.margin
        f.attrs['pol'] = tileset.pol
        
        # Store main data array
        f.create_dataset('data', data=tileset.data)

def read_tileset_hdf5(filepath):
    """
    Read a TileSet efficiently from an HDF5 file, verifying all class metadata.
    
    Parameters
    ----------
    filepath : str
        Path of the source HDF5 file.
        
    Returns
    -------
    TileSet
        A correctly initialized TileSet object recovering the saved metadata.
    """
    from .tileset import TileSet
    
    with h5py.File(filepath, 'r') as f:
        # Prevent silent extraction errors by strictly checking metadata
        required_attrs = ['nside', 'tile_nside', 'margin', 'pol']
        for attr in required_attrs:
            if attr not in f.attrs:
                raise ValueError(f"HDF5 file '{filepath}' is missing the required '{attr}' metadata.")
                
        nside = int(f.attrs['nside'])
        tile_nside = int(f.attrs['tile_nside'])
        margin = int(f.attrs['margin'])
        pol = bool(f.attrs['pol'])
        
        if 'data' not in f:
            raise ValueError(f"HDF5 file '{filepath}' does not contain the core 'data' dataset.")
            
        # Extract data explicitly into memory
        data = f['data'][:]
                
        return TileSet(data, nside, tile_nside, margin, pol=pol)

