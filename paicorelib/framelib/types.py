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
ArrayType = TypeVar("ArrayType", list[int], tuple[int, ...], np.ndarray)
BasicFrameArray = TypeVar(
    "BasicFrameArray", int, list[int], tuple[int, ...], NDArray[FRAME_DTYPE]
)
IntScalarType = TypeVar("IntScalarType", int, np.bool_, np.integer)
DataType = TypeVar("DataType", int, np.bool_, np.integer, np.ndarray)
DataArrayType = TypeVar(
    "DataArrayType", int, np.bool_, np.integer, list[int], tuple[int, ...], np.ndarray
)
