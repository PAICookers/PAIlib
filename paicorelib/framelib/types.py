import sys
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

FRAME_DTYPE: TypeAlias = np.uint64
FrameArrayType: TypeAlias = NDArray[FRAME_DTYPE]
PAYLOAD_DATA_DTYPE: TypeAlias = np.uint8  # in work frame type I
PayloadDataType: TypeAlias = NDArray[PAYLOAD_DATA_DTYPE]
LUT_DTYPE: TypeAlias = np.int8
LUTDataType: TypeAlias = NDArray[LUT_DTYPE]

ArrayType = TypeVar("ArrayType", list[int], tuple[int, ...], np.ndarray)
BasicFrameArray = TypeVar(
    "BasicFrameArray", int, list[int], tuple[int, ...], NDArray[FRAME_DTYPE]
)
IntScalarType = TypeVar("IntScalarType", int, np.bool, np.integer)
DataType = TypeVar("DataType", int, np.bool, np.integer, np.ndarray)
DataArrayType = TypeVar(
    "DataArrayType", int, np.bool, np.integer, list[int], tuple[int, ...], np.ndarray
)
