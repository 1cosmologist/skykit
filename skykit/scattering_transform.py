"""Scattering statistics of precomputed wavelet feature maps."""

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .feature_maps import FeatureMap, FeatureMapSet, open_feature_maps


@dataclass
class ScatteringStatistics:
    """Values and wavelet paths for unary or cross scattering statistics.

    ``values`` maps U1/U2 or cross keys such as U1_U2 to arrays. ``paths``
    maps each key to ``(paths1, paths2)``; paths2 is None for unary values.
    Each path is ``(j, theta)`` or ``(j1, theta1, j2, theta2)``.
    """

    values: dict
    paths: dict
    reduction: str
    single_tile: bool
    pol: bool

    def __getitem__(self, key):
        return self.values[key]

    def keys(self):
        return self.values.keys()

    def statistic(self, key, path1, path2=None, *, tile=None, stokes=None):
        """Fetch a statistic using its group and scale/orientation paths."""
        first, second = self.paths[key]
        indices = []
        if not self.single_tile:
            if tile is None:
                raise ValueError("tile is required for a tile set")
            indices.append(tile)
        if self.pol:
            if stokes is None:
                raise ValueError("stokes is required for polarized statistics")
            indices.append(stokes)
        indices.append(first.index(tuple(path1)))
        if second is not None:
            if path2 is None:
                raise ValueError("path2 is required for cross statistics")
            indices.append(second.index(tuple(path2)))
        return self.values[key][tuple(indices)]


def scattering_transform(feature_maps1, feature_maps2=None, *, operation1=None,
                         operation2=None, reduction="mean", path_batch_size=8):
    """Reduce wavelet features over tile pixels.

    Accept a FeatureMap, FeatureMapSet, or HDF5 path for each input. With one
    input, reduce ``operation1(U)`` for every feature path. With two inputs,
    return only cross statistics, reducing the pixelwise product
    ``operation1(U) * operation2(V)`` for every path pair. Operations are
    JAX-compatible callables that preserve map shape; None is identity.
    Reduction is the pixel mean or population variance. ``path_batch_size``
    bounds disk reads and intermediate products.
    """
    if reduction not in ("mean", "variance"):
        raise ValueError("reduction must be 'mean' or 'variance'")
    if not isinstance(path_batch_size, int) or path_batch_size < 1:
        raise ValueError("path_batch_size must be a positive integer")
    operation1 = operation1 or (lambda x: x)
    operation2 = operation2 or (lambda x: x)
    opened = []
    try:
        if isinstance(feature_maps1, (str, Path)):
            feature_maps1 = open_feature_maps(feature_maps1)
            opened.append(feature_maps1)
        if isinstance(feature_maps2, (str, Path)):
            feature_maps2 = open_feature_maps(feature_maps2)
            opened.append(feature_maps2)
        if not isinstance(feature_maps1, (FeatureMap, FeatureMapSet)):
            raise TypeError("feature_maps1 must be a FeatureMap or FeatureMapSet")
        if feature_maps2 is not None:
            if not isinstance(feature_maps2, (FeatureMap, FeatureMapSet)):
                raise TypeError("feature_maps2 must be a FeatureMap or FeatureMapSet")
            if type(feature_maps1) is not type(feature_maps2) or feature_maps1.pol != feature_maps2.pol:
                raise ValueError("feature sets must have matching tile layouts")
            a = next(iter(feature_maps1.maps.values()))
            b = next(iter(feature_maps2.maps.values()))
            if a.shape[:-3] != b.shape[:-3] or a.shape[-2:] != b.shape[-2:]:
                raise ValueError("feature sets must have matching tile and pixel shapes")
        return _compute(feature_maps1, feature_maps2, operation1, operation2,
                        reduction, path_batch_size)
    finally:
        for item in opened:
            item.close()


def _compute(first, second, operation1, operation2, reduction, path_batch_size):
    reduce = jnp.mean if reduction == "mean" else jnp.var
    values = {}
    paths = {}
    shape = next(iter(first.maps.values())).shape
    leading_shape, pixels = shape[:-3], shape[-2:]
    count = int(np.prod(leading_shape)) if leading_shape else 1

    def read(array, index, start, stop):
        leading = np.unravel_index(index, leading_shape) if leading_shape else ()
        return jnp.asarray(array[leading + (slice(start, stop), slice(None), slice(None))])

    for name1, maps1 in first.maps.items():
        n1 = len(first.paths[name1])
        groups = [(None, None)] if second is None else list(second.maps.items())
        for name2, maps2 in groups:
            n2 = 0 if maps2 is None else len(second.paths[name2])
            output = None
            for index in range(count):
                for start1 in range(0, n1, path_batch_size):
                    stop1 = min(start1 + path_batch_size, n1)
                    operated1 = operation1(read(maps1, index, start1, stop1))
                    if operated1.shape != (stop1 - start1,) + pixels:
                        raise ValueError("operation1 must preserve feature map shape")
                    if maps2 is None:
                        reduced = np.asarray(reduce(operated1, axis=(-2, -1)))
                        if output is None:
                            output = np.empty((count, n1), dtype=reduced.dtype)
                        output[index, start1:stop1] = reduced
                        continue
                    for start2 in range(0, n2, path_batch_size):
                        stop2 = min(start2 + path_batch_size, n2)
                        operated2 = operation2(read(maps2, index, start2, stop2))
                        if operated2.shape != (stop2 - start2,) + pixels:
                            raise ValueError("operation2 must preserve feature map shape")
                        reduced = np.asarray(reduce(operated1[:, None] * operated2[None, :],
                                                    axis=(-2, -1)))
                        if output is None:
                            output = np.empty((count, n1, n2), dtype=reduced.dtype)
                        output[index, start1:stop1, start2:stop2] = reduced
            key = name1 if name2 is None else f"{name1}_{name2}"
            if output is None:
                empty_shape = (count, n1) if maps2 is None else (count, n1, n2)
                output = np.empty(empty_shape, dtype=np.float32)
            values[key] = output.reshape(leading_shape + output.shape[1:])
            paths[key] = (first.paths[name1], None if name2 is None else second.paths[name2])
    return ScatteringStatistics(values, paths, reduction, first.single_tile, first.pol)
