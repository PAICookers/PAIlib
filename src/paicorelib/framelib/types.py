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
