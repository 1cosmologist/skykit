API reference
=============

This reference is generated from the public classes and functions in the
package docstrings. For the geometry and mathematical conventions, see the
:doc:`tiling` and :doc:`wavelets-scattering` guides.

HEALPix tiling
--------------

.. automodule:: skykit.healpix_tiling
   :members: HealpixTileProjector, healpix2tiles, tiles2healpix

Tile container
--------------

.. automodule:: skykit.tileset
   :members: TileSet

Tile utilities
--------------

.. automodule:: skykit.tile_utils
   :members: create_apodization_window, apply_apodization, write_tileset_hdf5, read_tileset_hdf5

Wavelet filters
---------------

.. automodule:: skykit.jax_wavelets
   :members: gabor_2d, morlet_2d, bump_2d, lowpass_2d, generate_filter_bank

Scattering transform
--------------------

.. automodule:: skykit.scattering_transform
   :members: Scattering2D

Plotting
--------

.. automodule:: skykit.tile_plotting
   :members: get_tile_wcs, plot_tile_flat, plot_tile_proj, plot_tile_flat_at, plot_tile_proj_at, plot_scattering_coefs, plot_scattering_tile
