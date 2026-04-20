import numpy as np
import healpy as hp

class TileSet:
    """
    Container for overlapping square tiles extracted from a HEALPix map.

    Tiles are in canonical order: face slowest, then tx, then ty.
    Linear index: ``idx = (face * n_subtiles + tx) * n_subtiles + ty``.

    Attributes
    ----------
    data : ndarray
        If scalar: shape ``(n_tiles, tile_full, tile_full)``.
        If pol: shape ``(n_tiles, 2, tile_full, tile_full)`` where
        axis 1 is [Q, U] in the tile-centre frame.
    pol : bool
        Whether this TileSet holds polarisation data.
    nside, tile_nside, margin : int
    n_subtiles, n_tiles, tile_full : int
    """

    def __init__(self, data, nside, tile_nside, margin, pol=False):
        self.data = data
        self.pol = pol
        self.nside = nside
        self.tile_nside = tile_nside
        self.margin = margin
        self.n_subtiles = nside // tile_nside
        self.n_tiles = data.shape[0]
        self.tile_full = tile_nside + 2 * margin
        self.faces = np.arange(12, dtype=np.int32)
        self.tx_values = np.arange(self.n_subtiles, dtype=np.int32)
        self.ty_values = np.arange(self.n_subtiles, dtype=np.int32)

    def tile_index(self, face, tx, ty):
        """Linear index into ``self.data`` for tile (face, tx, ty)."""
        return (face * self.n_subtiles + tx) * self.n_subtiles + ty

    def get_tile(self, face, tx, ty):
        """Return full tile data (interior + border)."""
        idx = self.tile_index(face, tx, ty)
        return self.data[idx]

    def get_interior(self, face, tx, ty):
        """Return only the interior of the tile (no border)."""
        m = self.margin
        s = self.tile_nside
        idx = self.tile_index(face, tx, ty)
        if self.pol:
            return self.data[idx, :, m: m + s, m: m + s]
        else:
            return self.data[idx, m: m + s, m: m + s]

    def tile_center(self, face, tx, ty):
        """
        Angular position (lon, lat) in degrees of the tile centre.
        """
        ix = tx * self.tile_nside + self.tile_nside // 2
        iy = ty * self.tile_nside + self.tile_nside // 2
        ipix = hp.xyf2pix(self.nside, ix, iy, face, nest=True)
        theta, phi = hp.pix2ang(self.nside, ipix, nest=True)
        lon = np.degrees(phi) % 360.0
        lat = 90.0 - np.degrees(theta)
        return float(lon), float(lat)

    def tile_containing(self, lon, lat):
        """
        Return (face, tx, ty) of the tile whose interior contains
        the given sky position.
        """
        theta = np.radians(90.0 - lat)
        phi = np.radians(lon) % (2.0 * np.pi)
        if not (0.0 <= theta <= np.pi):
            raise ValueError(f"Latitude {lat} out of range [-90, 90]")
        ipix = hp.ang2pix(self.nside, theta, phi, nest=True)
        ix, iy, face = hp.pix2xyf(self.nside, ipix, nest=True)
        tx = int(ix) // self.tile_nside
        ty = int(iy) // self.tile_nside
        return int(face), tx, ty

    def iter_tiles(self):
        """Iterate in canonical order, yielding (face, tx, ty, tile_data)."""
        for face in self.faces:
            for tx in self.tx_values:
                for ty in self.ty_values:
                    yield (int(face), int(tx), int(ty),
                           self.get_tile(face, tx, ty))
                    
    def __repr__(self):
        kind = "pol" if self.pol else "scalar"
        return (
            f"TileSet({kind}, nside={self.nside}, "
            f"tile_nside={self.tile_nside}, "
            f"margin={self.margin}, n_tiles={self.n_tiles}, "
            f"tile_full={self.tile_full}, "
            f"data.shape={self.data.shape})"
        )


