from typing import TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray

FRAME_DTYPE: TypeAlias = np.uint64
FrameArrayType: TypeAlias = NDArray[FRAME_DTYPE]
PAYLOAD_DATA_DTYPE: TypeAlias = np.uint8  # in work frame type I
PayloadDataType: TypeAlias = NDArray[PAYLOAD_DATA_DTYPE]
LUT_DTYPE: TypeAlias = np.int8
LUTDataType: TypeAlias = NDArray[LUT_DTYPE]

BasicFrameArray = TypeVar(
    "BasicFrameArray", int, list[int], tuple[int, ...], NDArray[FRAME_DTYPE]
)
IntScalarType = TypeVar("IntScalarType", int, np.bool, np.integer)
DataType = TypeVar("DataType", int, np.bool, np.integer, np.ndarray)
DataArrayType = TypeVar(
    "DataArrayType", int, np.bool, np.integer, list[int], tuple[int, ...], np.ndarray
)
