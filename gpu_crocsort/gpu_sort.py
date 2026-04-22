"""
gpu_sort — PyArrow-compatible GPU sort using ctypes + libgpusort.so

Provides drop-in replacements for pyarrow.compute.sort_indices / take:
    sort_indices(array_or_table, sort_keys=None, verbose=False)
    sort_table(table, sort_keys, verbose=False)
    sort_array(array, descending=False, verbose=False)
"""

import ctypes
import os
import struct
import numpy as np
import pyarrow as pa

# ---------------------------------------------------------------------------
# Load the shared library from the same directory as this module
# ---------------------------------------------------------------------------
_dir = os.path.dirname(os.path.abspath(__file__))
_lib_path = os.path.join(_dir, "libgpusort.so")
_lib = ctypes.CDLL(_lib_path)

# int gpu_sort_keys(const uint8_t* keys, uint32_t key_size, uint32_t key_stride,
#                   uint64_t num_records, uint32_t* perm_out);
_lib.gpu_sort_keys.argtypes = [
    ctypes.c_void_p,   # keys
    ctypes.c_uint32,   # key_size
    ctypes.c_uint32,   # key_stride
    ctypes.c_uint64,   # num_records
    ctypes.c_void_p,   # perm_out
]
_lib.gpu_sort_keys.restype = ctypes.c_int

# int gpu_sort_available(void);
_lib.gpu_sort_available.argtypes = []
_lib.gpu_sort_available.restype = ctypes.c_int

# GpuSortTiming struct
class GpuSortTiming(ctypes.Structure):
    _fields_ = [
        ("total_ms", ctypes.c_double),
        ("upload_ms", ctypes.c_double),
        ("gpu_sort_ms", ctypes.c_double),
        ("download_ms", ctypes.c_double),
        ("gather_ms", ctypes.c_double),
        ("fixup_ms", ctypes.c_double),
        ("num_fixup_groups", ctypes.c_uint64),
        ("num_fixup_records", ctypes.c_uint64),
    ]

_lib.gpu_sort_get_timing.argtypes = [ctypes.POINTER(GpuSortTiming)]
_lib.gpu_sort_get_timing.restype = None

# ---------------------------------------------------------------------------
# Byte-comparable key encoding
# ---------------------------------------------------------------------------
# For each supported type we produce a fixed-width big-endian byte string
# such that memcmp gives the correct sort order.
#
# Signed integers:  flip sign bit, then big-endian
# Unsigned integers: big-endian
# Floats:  if positive, flip sign bit; if negative, flip all bits (IEEE trick)
# date32:  same as int32

_SUPPORTED_TYPES = {
    pa.int8():    (1, "signed"),
    pa.int16():   (2, "signed"),
    pa.int32():   (4, "signed"),
    pa.int64():   (8, "signed"),
    pa.uint8():   (1, "unsigned"),
    pa.uint16():  (2, "unsigned"),
    pa.uint32():  (4, "unsigned"),
    pa.uint64():  (8, "unsigned"),
    pa.float32(): (4, "float"),
    pa.float64(): (8, "float"),
    pa.date32():  (4, "signed"),
}


def _encode_column(arr: pa.Array) -> np.ndarray:
    """Encode a pyarrow Array into byte-comparable keys.

    Returns an ndarray of shape (n, key_width) with dtype=uint8.
    """
    typ = arr.type
    if typ not in _SUPPORTED_TYPES:
        raise TypeError(f"Unsupported type for GPU sort: {typ}. "
                        f"Supported: {list(_SUPPORTED_TYPES.keys())}")

    width, kind = _SUPPORTED_TYPES[typ]

    # Convert to numpy — handle nulls by filling with 0 (sorts first)
    if typ == pa.date32():
        np_arr = arr.cast(pa.int32()).to_numpy(zero_copy_only=False)
    else:
        np_arr = arr.to_numpy(zero_copy_only=False)

    if kind == "signed":
        # Reinterpret as unsigned of same width, flip sign bit, big-endian bytes
        if width == 1:
            u = np_arr.astype(np.int8).view(np.uint8)
            u = u ^ np.uint8(0x80)
            return u.reshape(-1, 1)
        elif width == 2:
            u = np_arr.astype(np.int16).view(np.uint16)
            u = u ^ np.uint16(0x8000)
            # Convert to big-endian
            u = u.astype(">u2")
            return u.view(np.uint8).reshape(-1, 2)
        elif width == 4:
            u = np_arr.astype(np.int32).view(np.uint32)
            u = u ^ np.uint32(0x80000000)
            u = u.astype(">u4")
            return u.view(np.uint8).reshape(-1, 4)
        elif width == 8:
            u = np_arr.astype(np.int64).view(np.uint64)
            u = u ^ np.uint64(0x8000000000000000)
            u = u.astype(">u8")
            return u.view(np.uint8).reshape(-1, 8)

    elif kind == "unsigned":
        if width == 1:
            return np_arr.astype(np.uint8).reshape(-1, 1)
        elif width == 2:
            return np_arr.astype(">u2").view(np.uint8).reshape(-1, 2)
        elif width == 4:
            return np_arr.astype(">u4").view(np.uint8).reshape(-1, 4)
        elif width == 8:
            return np_arr.astype(">u8").view(np.uint8).reshape(-1, 8)

    elif kind == "float":
        # IEEE float encoding: if sign bit set, flip all bits; else flip sign bit only
        if width == 4:
            u = np_arr.astype(np.float32).view(np.uint32)
            mask = np.where(u & np.uint32(0x80000000),
                            np.uint32(0xFFFFFFFF),
                            np.uint32(0x80000000))
            u = u ^ mask
            u = u.astype(">u4")
            return u.view(np.uint8).reshape(-1, 4)
        elif width == 8:
            u = np_arr.astype(np.float64).view(np.uint64)
            mask = np.where(u & np.uint64(0x8000000000000000),
                            np.uint64(0xFFFFFFFFFFFFFFFF),
                            np.uint64(0x8000000000000000))
            u = u ^ mask
            u = u.astype(">u8")
            return u.view(np.uint8).reshape(-1, 8)

    raise TypeError(f"Unhandled encoding for {typ}")


def _build_sort_keys(array_or_table, sort_keys=None):
    """Build a contiguous byte-comparable key buffer.

    Returns (keys_buf: np.ndarray[uint8] with shape (n, stride), key_size, n).
    """
    if isinstance(array_or_table, pa.Array) or isinstance(array_or_table, pa.ChunkedArray):
        if isinstance(array_or_table, pa.ChunkedArray):
            array_or_table = array_or_table.combine_chunks()
        encoded = _encode_column(array_or_table)
        return encoded, encoded.shape[1], len(array_or_table)

    if not isinstance(array_or_table, pa.Table):
        raise TypeError(f"Expected pa.Array, pa.ChunkedArray, or pa.Table, got {type(array_or_table)}")

    table = array_or_table
    if sort_keys is None:
        raise ValueError("sort_keys required when sorting a Table")

    n = len(table)
    parts = []
    for col_name, order in sort_keys:
        col = table.column(col_name)
        if isinstance(col, pa.ChunkedArray):
            col = col.combine_chunks()
        encoded = _encode_column(col)
        if order == "descending":
            encoded = ~encoded  # bitwise NOT for descending
        parts.append(encoded)

    # Concatenate columns horizontally
    keys = np.concatenate(parts, axis=1)
    # Make contiguous
    keys = np.ascontiguousarray(keys)
    return keys, keys.shape[1], n


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sort_indices(array_or_table, sort_keys=None, verbose=False):
    """Return a UInt32Array of sorted indices (like pyarrow.compute.sort_indices).

    Parameters
    ----------
    array_or_table : pa.Array or pa.Table
        Data to sort.
    sort_keys : list of (column_name, order) or None
        For tables: list of (col, "ascending"/"descending").
        For arrays: None (ascending) or ignored.
    verbose : bool
        Print GPU timing breakdown.

    Returns
    -------
    pa.UInt32Array
    """
    keys, key_size, n = _build_sort_keys(array_or_table, sort_keys)

    # Ensure C-contiguous uint8
    keys = np.ascontiguousarray(keys, dtype=np.uint8)
    key_stride = keys.strides[0]  # bytes per row

    perm = np.empty(n, dtype=np.uint32)

    rc = _lib.gpu_sort_keys(
        keys.ctypes.data,
        ctypes.c_uint32(key_size),
        ctypes.c_uint32(key_stride),
        ctypes.c_uint64(n),
        perm.ctypes.data,
    )
    if rc != 0:
        raise RuntimeError(f"gpu_sort_keys failed with code {rc}")

    if verbose:
        t = GpuSortTiming()
        _lib.gpu_sort_get_timing(ctypes.byref(t))
        print(f"GPU sort timing ({n:,} rows, {key_size}B keys):")
        print(f"  total:    {t.total_ms:8.1f} ms")
        print(f"  upload:   {t.upload_ms:8.1f} ms")
        print(f"  gpu_sort: {t.gpu_sort_ms:8.1f} ms")
        print(f"  download: {t.download_ms:8.1f} ms")
        print(f"  fixup:    {t.fixup_ms:8.1f} ms  "
              f"({t.num_fixup_groups} groups, {t.num_fixup_records} records)")

    return pa.array(perm, type=pa.uint32())


def sort_table(table, sort_keys, verbose=False):
    """Sort a pyarrow Table, returning a new sorted Table.

    Parameters
    ----------
    table : pa.Table
    sort_keys : list of (column_name, "ascending"/"descending")
    verbose : bool

    Returns
    -------
    pa.Table
    """
    indices = sort_indices(table, sort_keys=sort_keys, verbose=verbose)
    return table.take(indices)


def sort_array(array, descending=False, verbose=False):
    """Sort a pyarrow Array, returning a new sorted Array.

    Parameters
    ----------
    array : pa.Array
    descending : bool
    verbose : bool

    Returns
    -------
    pa.Array
    """
    if descending:
        # Build keys with descending encoding
        if isinstance(array, pa.ChunkedArray):
            array = array.combine_chunks()
        encoded = _encode_column(array)
        encoded = ~encoded
        encoded = np.ascontiguousarray(encoded, dtype=np.uint8)
        n = len(array)
        key_size = encoded.shape[1]
        key_stride = encoded.strides[0]
        perm = np.empty(n, dtype=np.uint32)
        rc = _lib.gpu_sort_keys(
            encoded.ctypes.data,
            ctypes.c_uint32(key_size),
            ctypes.c_uint32(key_stride),
            ctypes.c_uint64(n),
            perm.ctypes.data,
        )
        if rc != 0:
            raise RuntimeError(f"gpu_sort_keys failed with code {rc}")
        if verbose:
            t = GpuSortTiming()
            _lib.gpu_sort_get_timing(ctypes.byref(t))
            print(f"GPU sort timing ({n:,} rows, {key_size}B keys, descending):")
            print(f"  total:    {t.total_ms:8.1f} ms")
            print(f"  upload:   {t.upload_ms:8.1f} ms")
            print(f"  gpu_sort: {t.gpu_sort_ms:8.1f} ms")
            print(f"  download: {t.download_ms:8.1f} ms")
            print(f"  fixup:    {t.fixup_ms:8.1f} ms")
        indices = pa.array(perm, type=pa.uint32())
    else:
        indices = sort_indices(array, verbose=verbose)
    return array.take(indices)


def is_available():
    """Return True if a CUDA GPU is available."""
    return bool(_lib.gpu_sort_available())
