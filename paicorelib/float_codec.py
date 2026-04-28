"""Floating-point codec helpers for PAICORE hardware formats.

The module is organized around actual usage:

* ``BF16Param`` / ``FP32Param`` for Pydantic model fields.
* ``cast_*_scalar`` helpers for fp32 scalar carrier values.
* ``pack_*`` helpers for hardware bit patterns used by frame generation.
* ``pack_bf16_payload_bits`` for frame payload compatibility when integer
  inputs are already encoded BF16 payloads.
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
    return np.float32(value).view(_U32_DTYPE).item()


def pack_fp32_payload_bits(values: ArrayLike) -> NDArray[np.uint32]:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.floating):
        return np.ascontiguousarray(arr, dtype=np.float32).view(_U32_DTYPE)

    return np.ascontiguousarray(arr, _U32_DTYPE)


def pack_bf16_payload_bits(values: ArrayLike) -> NDArray[np.uint16]:
    """Return frame payload bits for BF16 fields.

    Floating-point inputs are packed numerically. Integer inputs are treated as
    already-encoded BF16 payloads.
    """
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.floating):
        return (
            np.ascontiguousarray(arr, dtype=np.float32).view(_U32_DTYPE) >> 16
        ).astype(_U16_DTYPE)

    return np.ascontiguousarray(arr, _U16_DTYPE)


def pack_bf16_scalar_bits(value: SupportsFloat) -> int:
    return pack_fp32_scalar_bits(value) >> 16


def _require_finite(value: SupportsFloat) -> float:
    value_f = float(value)
    if not math.isfinite(value_f):
        raise ValueError(f"floating-point parameter must be finite, but got {value_f}")

    return value_f


def cast_fp32_scalar(value: SupportsFloat) -> float:
    return np.float32(value).item()


def cast_bf16_scalar(value: SupportsFloat) -> float:
    return cast_fp32_scalar(value)


FP32Param = Annotated[
    float, AfterValidator(_require_finite), AfterValidator(cast_fp32_scalar)
]

BF16Param = Annotated[
    float, AfterValidator(_require_finite), AfterValidator(cast_bf16_scalar)
]
