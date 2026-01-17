from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

FRAME_DTYPE = np.uint64
PAYLOAD_DATA_DTYPE = np.uint8  # in work frame type I
LUT_DTYPE = np.int8
FrameArrayType = NDArray[FRAME_DTYPE]
PayloadDataType = NDArray[PAYLOAD_DATA_DTYPE]
LUTDataType = NDArray[LUT_DTYPE]

LUT_POTENTIAL_DTYPE = np.int32
LUTPotentialType = NDArray[LUT_POTENTIAL_DTYPE]
LUT_ACTIVATION_DTYPE = np.int8 | np.uint8
LUTActivationType = NDArray[LUT_ACTIVATION_DTYPE]

FrameArrayLike = TypeVar(
    "FrameArrayLike", int, list[int], tuple[int, ...], NDArray[FRAME_DTYPE]
)
IntScalarType = TypeVar("IntScalarType", int, bool, np.integer)
DataType = TypeVar("DataType", int, bool, np.integer, np.ndarray)
