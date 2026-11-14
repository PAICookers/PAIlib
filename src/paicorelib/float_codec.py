"""Numeric carriers and raw payload bits for PAICORE frame generation.

BF16Param/FP32Param check input finiteness, then retain float32 numeric carriers;
BF16 truncation happens only at packing. A finite value can overflow that cast.
Scalar packers interpret values numerically; array packers treat floating dtypes
numerically and integer dtypes as raw codes. Bit views target little-endian hosts;
returned words are not a serialized transport byte stream.
"""

import math
from typing import Annotated, SupportsFloat

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import AfterValidator

__all__ = [
    "BF16Param",
    "FP32Param",
    "cast_bf16_scalar",
    "cast_fp32_scalar",
    "pack_bf16_payload_bits",
    "pack_fp32_payload_bits",
    "pack_bf16_scalar_bits",
    "pack_fp32_scalar_bits",
]

_U16_DTYPE = np.dtype("<u2")  # little-endian uint16 (for BF16 payloads)
_U32_DTYPE = np.dtype("<u4")  # little-endian uint32


def pack_fp32_scalar_bits(value: SupportsFloat) -> int:
    """Return a numeric scalar's float32 bits, e.g. ``1.0 -> 0x3f800000``.

    NumPy conversion/rounding applies; this helper permits NaN and infinity.
    """
    return np.float32(value).view(_U32_DTYPE).item()


def pack_fp32_payload_bits(values: ArrayLike) -> NDArray[np.uint32]:
    """Pack floats numerically as FP32; cast non-floating input to raw uint32.

    Non-scalar shapes are preserved; scalars become (1,). Output is contiguous,
    may share storage, and has no additional finite/range validation.
    """
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.floating):
        return np.ascontiguousarray(arr, dtype=np.float32).view(_U32_DTYPE)

    return np.ascontiguousarray(arr, _U32_DTYPE)


def pack_bf16_payload_bits(values: ArrayLike) -> NDArray[np.uint16]:
    """Pack floats by retaining FP32's high 16 bits; cast integers to raw uint16.

    BF16 precision is truncated, not rounded: ``[1.0] -> [0x3f80]`` whereas
    ``[1] -> [1]``. Shapes follow the FP32 packer; raw casts may narrow bits.
    """
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.floating):
        return (
            np.ascontiguousarray(arr, dtype=np.float32).view(_U32_DTYPE) >> 16
        ).astype(_U16_DTYPE)

    return np.ascontiguousarray(arr, _U16_DTYPE)


def pack_bf16_scalar_bits(value: SupportsFloat) -> int:
    """Return a numeric scalar's high 16 FP32 bits, without BF16 rounding."""
    return pack_fp32_scalar_bits(value) >> 16


def _require_finite(value: SupportsFloat) -> float:
    value_f = float(value)
    if not math.isfinite(value_f):
        raise ValueError(f"floating-point parameter must be finite, but got {value_f}")

    return value_f


def cast_fp32_scalar(value: SupportsFloat) -> float:
    """Round through NumPy float32 to a Python float; no finite check is added."""
    return np.float32(value).item()


def cast_bf16_scalar(value: SupportsFloat) -> float:
    """Retain a float32 numeric carrier; BF16 truncation occurs only at packing."""
    return cast_fp32_scalar(value)


FP32Param = Annotated[
    float, AfterValidator(_require_finite), AfterValidator(cast_fp32_scalar)
]

BF16Param = Annotated[
    float, AfterValidator(_require_finite), AfterValidator(cast_bf16_scalar)
]
