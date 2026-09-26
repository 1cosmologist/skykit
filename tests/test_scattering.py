import unittest
import numpy as np

from skykit import Scattering2D, TileSet, generate_filter_bank

class TestScattering2D(unittest.TestCase):
    def setUp(self):
        # Set up a small 32x32 image size for fast testing
        self.M = 32
        self.N = 32
        self.filter_bank = generate_filter_bank(self.M, self.N, J=2, L=4)
        self.scatter = Scattering2D(self.filter_bank, max_order=2)

    def test_initialization(self):
        """Check dimensions inferred from the supplied filter bank."""
        self.assertEqual(self.scatter.M, self.M)
        self.assertEqual(self.scatter.N, self.N)
        self.assertEqual(self.scatter.J, 2)
        self.assertEqual(self.scatter.L, 4)
        
        # Check that psi filters are generated
        self.assertTrue(self.scatter.psi_vals.shape[0] > 0)
        self.assertEqual(self.scatter.psi_vals.shape[1:], (self.M, self.N))

    def test_scattering_averages_and_optional_feature_maps(self):
        x = np.random.default_rng(8).normal(size=(self.M, self.N))
        result = self.scatter.transform_tile(x, return_feature_maps=True)
        phi = np.asarray(self.filter_bank['phi']['val'])
        fft_x = np.fft.fft2(x)
        psi1 = np.asarray(self.filter_bank['psi'][0]['val'])
        psi2 = np.asarray(self.filter_bank['psi'][4]['val'])

        u1 = np.abs(np.fft.ifft2(fft_x * psi1))
        s1 = np.real(np.fft.ifft2(np.fft.fft2(u1) * phi))
        u2 = np.abs(np.fft.ifft2(np.fft.fft2(u1) * psi2))
        s2 = np.real(np.fft.ifft2(np.fft.fft2(u2) * phi))

        np.testing.assert_allclose(result['U1'][0], u1, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(result['S1'][0], s1, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(result['U2'][0], u2, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(result['S2'][0], s2, rtol=2e-5, atol=2e-6)
        self.assertGreater(np.max(np.abs(result['S1'][0] - result['U1'][0])), 0.01)

        global_result = self.scatter.transform_tile(x, spatial_average=True)
        self.assertEqual(global_result['S0'].shape, ())
        self.assertEqual(global_result['S1'].shape, (8,))
        self.assertEqual(global_result['S2'].shape, (16,))
        self.assertNotIn('U1', global_result)
        for key in ('S0', 'S1', 'S2'):
            np.testing.assert_allclose(global_result[key], result[key].mean(axis=(-2, -1)),
                                       rtol=2e-5, atol=2e-6)

    def test_tileset_global_shape_and_feature_maps(self):
        x = np.random.default_rng(9).normal(size=(12, 2, self.M, self.N))
        tileset = TileSet(x, nside=self.M, tile_nside=self.M, margin=0, pol=True)
        result = self.scatter.transform_tileset(
            tileset, spatial_average=True, return_feature_maps=True)
        self.assertEqual(result['S0'].shape, (12, 2))
        self.assertEqual(result['S1'].shape, (12, 2, 8))
        self.assertEqual(result['S2'].shape, (12, 2, 16))
        self.assertEqual(result['U1'].shape, (12, 2, 8, self.M, self.N))
        self.assertEqual(result['U2'].shape, (12, 2, 16, self.M, self.N))
        single = self.scatter.transform_tile(x[0], spatial_average=True)
        for key in ('S0', 'S1', 'S2'):
            np.testing.assert_allclose(result[key][0], single[key], rtol=2e-5, atol=2e-6)

    def test_single_scale_has_empty_second_order(self):
        scatter = Scattering2D(generate_filter_bank(16, 16, J=1, L=2))
        result = scatter.transform_tile(
            np.ones((16, 16)), spatial_average=True, return_feature_maps=True)
        self.assertEqual(result['S2'].shape, (0,))
        self.assertEqual(result['U2'].shape, (0, 16, 16))

if __name__ == '__main__':
    unittest.main()
