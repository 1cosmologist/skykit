import unittest
import numpy as np
import jax.numpy as jnp

from skykit import Scattering2D

class TestScattering2D(unittest.TestCase):
    def setUp(self):
        # Set up a small 32x32 image size for fast testing
        self.M = 32
        self.N = 32
        self.scatter = Scattering2D(self.M, self.N, J=2, L=4, max_order=2)

    def test_initialization(self):
        """Test if the Scattering2D object initializes correctly and filter bank is generated."""
        self.assertEqual(self.scatter.M, self.M)
        self.assertEqual(self.scatter.N, self.N)
        self.assertEqual(self.scatter.J, 2)
        self.assertEqual(self.scatter.L, 4)
        
        # Check that psi filters are generated
        self.assertTrue(self.scatter.psi_vals.shape[0] > 0)
        self.assertEqual(self.scatter.psi_vals.shape[1:], (self.M, self.N))

if __name__ == '__main__':
    unittest.main()
