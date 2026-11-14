"""NumPy carrier dtypes and static annotations; aliases do not validate values.

FRAME_DTYPE/FrameArrayType represent uint64 words, without transport byte order.
PAYLOAD_DATA_DTYPE is raw uint8; VOLTAGE_DTYPE is offline int32, not online FP32.
Legacy LUT_DTYPE is int8; V2 uses int32 LUT_POTENTIAL_DTYPE and int8/uint8
LUT_ACTIVATION_DTYPE, with corresponding array aliases. FrameArrayLike accepts
integer words, iterables and arrays for normalization by frame_array2np.
IntScalarType/DataType are generic integer/bool carriers, the latter also arrays.
"""

from collections.abc import Iterable
from typing import TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray

FRAME_DTYPE = np.uint64
PAYLOAD_DATA_DTYPE = np.uint8  # in work frame type I
LUT_DTYPE = np.int8
VOLTAGE_DTYPE = np.int32
FrameArrayType = NDArray[FRAME_DTYPE]
PayloadDataType = NDArray[PAYLOAD_DATA_DTYPE]
LUTDataType = NDArray[LUT_DTYPE]
VoltageDataType = NDArray[VOLTAGE_DTYPE]

LUT_POTENTIAL_DTYPE = np.int32
LUTPotentialType = NDArray[LUT_POTENTIAL_DTYPE]
LUT_ACTIVATION_DTYPE = np.int8 | np.uint8
LUTActivationType = NDArray[LUT_ACTIVATION_DTYPE]

FrameScalarLike: TypeAlias = int | np.integer
FrameArrayLike: TypeAlias = (
    FrameScalarLike | Iterable[FrameScalarLike] | NDArray[FRAME_DTYPE]
)
IntScalarType = TypeVar("IntScalarType", int, bool, np.integer)
DataType = TypeVar("DataType", int, bool, np.integer, np.ndarray)
