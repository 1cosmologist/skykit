import unittest
import numpy as np
import healpy as hp
import tempfile
import os

from skykit.healpix_tiling import HealpixTileProjector, healpix2tiles, tiles2healpix
from skykit.tile_utils import write_tileset_hdf5, read_tileset_hdf5
from skykit.tileset import TileSet

class TestHealpixTiling(unittest.TestCase):
    def setUp(self):
        self.nside = 16
        self.tile_nside = 8
        self.margin = 2
        self.npix = 12 * self.nside * self.nside
        
        # Create a dummy scalar map
        np.random.seed(42)
        self.scalar_map = np.random.randn(self.npix)
        
        # Create a dummy polarization map (Q, U)
        self.pol_map = np.random.randn(2, self.npix)

    def test_healpix_tile_projector_scalar(self):
        projector = HealpixTileProjector(self.nside, self.tile_nside, self.margin, pol=False)
        tileset = projector.map2tiles(self.scalar_map)
        
        self.assertEqual(tileset.nside, self.nside)
        self.assertEqual(tileset.tile_nside, self.tile_nside)
        self.assertEqual(tileset.margin, self.margin)
        self.assertFalse(tileset.pol)
        self.assertTrue(not hasattr(tileset, 'psi') or tileset.psi is None)
        
        reconstructed = projector.tiles2map(tileset)
        np.testing.assert_allclose(self.scalar_map, reconstructed, rtol=1e-10)

    def test_healpix_tile_projector_pol(self):
        projector = HealpixTileProjector(self.nside, self.tile_nside, self.margin, pol=True)
        tileset = projector.map2tiles(self.pol_map)
        
        self.assertEqual(tileset.data.shape[1], 2)  # Q, U
        self.assertTrue(tileset.pol)
        # psi is no longer stored explicitly on the class permanently!
        self.assertTrue(not hasattr(tileset, 'psi') or tileset.psi is None)
        
        reconstructed = projector.tiles2map(tileset)
        np.testing.assert_allclose(self.pol_map, reconstructed, rtol=1e-10)

    def test_hdf5_io(self):
        projector = HealpixTileProjector(self.nside, self.tile_nside, self.margin, pol=True)
        tileset = projector.map2tiles(self.pol_map)
        
        # Use temp file
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            write_tileset_hdf5(tileset, tmp_path)
            recovered = read_tileset_hdf5(tmp_path)
            
            self.assertEqual(tileset.nside, recovered.nside)
            self.assertEqual(tileset.tile_nside, recovered.tile_nside)
            self.assertEqual(tileset.margin, recovered.margin)
            self.assertEqual(tileset.pol, recovered.pol)
            np.testing.assert_allclose(tileset.data, recovered.data)
            
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == '__main__':
    unittest.main()
