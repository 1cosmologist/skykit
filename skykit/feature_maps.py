"""Wavelet feature maps, path metadata, and HDF5 storage."""

from pathlib import Path

import h5py
import numpy as np


_PATH_WIDTH = {"U1": 2, "U2": 4}


def _validate(maps, paths, single_tile, pol):
    if not maps or set(maps) != set(paths) or not set(maps) <= set(_PATH_WIDTH):
        raise ValueError("maps and paths must contain matching U1/U2 keys")
    rank = 3 + int(pol) + int(not single_tile)
    leading = None
    pixels = None
    for key, data in maps.items():
        if len(data.shape) != rank:
            raise ValueError(f"{key} has the wrong rank for this feature-map type")
        if data.shape[-3] != len(paths[key]):
            raise ValueError(f"{key} path count does not match its maps")
        if any(len(path) != _PATH_WIDTH[key] for path in paths[key]):
            raise ValueError(f"{key} paths must have {_PATH_WIDTH[key]} entries")
        if leading is None:
            leading, pixels = data.shape[:-3], data.shape[-2:]
        elif data.shape[:-3] != leading or data.shape[-2:] != pixels:
            raise ValueError("feature groups must share tile and pixel dimensions")


def _create_map_dataset(handle, key, shape, dtype, single_tile, pol):
    path_axis = len(shape) - 3
    chunks = None
    if all(shape):
        chunks = tuple(1 if axis < path_axis else min(8, size) if axis == path_axis
                       else size for axis, size in enumerate(shape))
    dataset = handle.create_dataset(key, shape=shape, dtype=dtype, chunks=chunks)
    labels = ([] if single_tile else ["tile"]) + (["stokes"] if pol else [])
    for axis, label in enumerate(labels + ["path", "x", "y"]):
        dataset.dims[axis].label = label
    return dataset


class _FeatureBase:
    """Common mapping and I/O behavior for one tile or a tileset."""

    single_tile = None

    def __init__(self, maps, paths, *, pol=False, metadata=None, _handle=None):
        self.maps = dict(maps)
        self.paths = {key: tuple(tuple(path) for path in group)
                      for key, group in paths.items()}
        self.pol = bool(pol)
        self.metadata = dict(metadata or {})
        self._handle = _handle
        _validate(self.maps, self.paths, self.single_tile, self.pol)

    def __getitem__(self, key):
        return self.maps[key]

    def keys(self):
        return self.maps.keys()

    def path_index(self, key, path):
        """Return the feature axis index for a scale/orientation path."""
        return self.paths[key].index(tuple(path))

    def feature(self, key, path, *, tile=None, stokes=None):
        """Fetch one feature map by its wavelet path and optional tile/component."""
        index = []
        if not self.single_tile:
            if tile is None:
                raise ValueError("tile is required for a FeatureMapSet")
            index.append(tile)
        if self.pol:
            if stokes is None:
                raise ValueError("stokes is required for polarized features")
            index.append(stokes)
        index.append(self.path_index(key, path))
        return self.maps[key][tuple(index)]

    def to_hdf5(self, filepath):
        """Save feature maps and scale/orientation paths in one HDF5 file."""
        filepath = Path(filepath)
        if self._handle is not None and Path(self._handle.filename).resolve() == filepath.resolve():
            raise ValueError("cannot overwrite an open feature-map file")
        with h5py.File(filepath, "w") as handle:
            handle.attrs["single_tile"] = self.single_tile
            handle.attrs["pol"] = self.pol
            for key, value in self.metadata.items():
                handle.attrs[key] = value
            for key, source in self.maps.items():
                paths = np.asarray(self.paths[key], dtype=np.float64).reshape(-1, _PATH_WIDTH[key])
                handle.create_dataset(f"{key}_paths", data=paths)
                target = _create_map_dataset(handle, key, source.shape, source.dtype,
                                             self.single_tile, self.pol)
                if self.single_tile:
                    target[...] = source
                else:
                    for tile in range(source.shape[0]):
                        target[tile] = source[tile]
        return filepath

    @classmethod
    def from_hdf5(cls, filepath):
        """Open feature maps lazily. Close the result or use a ``with`` block."""
        handle = h5py.File(filepath, "r")
        try:
            single_tile = bool(handle.attrs["single_tile"])
            target_cls = FeatureMap if single_tile else FeatureMapSet
            if cls is not _FeatureBase and cls is not target_cls:
                raise ValueError("file contains a different feature-map type")
            maps = {key: handle[key] for key in _PATH_WIDTH if key in handle}
            paths = {}
            for key in maps:
                raw = handle[f"{key}_paths"][:]
                paths[key] = tuple(tuple(int(row[i]) if i % 2 == 0 else float(row[i])
                                         for i in range(raw.shape[1])) for row in raw)
            metadata = {key: handle.attrs[key] for key in handle.attrs
                        if key not in ("single_tile", "pol")}
            return target_cls(maps, paths, pol=bool(handle.attrs["pol"]),
                              metadata=metadata, _handle=handle)
        except Exception:
            handle.close()
            raise

    def close(self):
        """Close the backing HDF5 file, if present."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class FeatureMap(_FeatureBase):
    """Feature maps for one scalar or polarized tile."""

    single_tile = True


class FeatureMapSet(_FeatureBase):
    """Feature maps for a scalar or polarized TileSet, in tile order."""

    single_tile = False


def open_feature_maps(filepath):
    """Open an HDF5 file as a lazy :class:`FeatureMap` or :class:`FeatureMapSet`."""
    return _FeatureBase.from_hdf5(filepath)
