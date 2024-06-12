import sys
import warnings
from functools import wraps
from pathlib import Path
from typing import Any, Optional

import numpy as np
from pydantic import TypeAdapter

from .frame_defs import FrameFormat as FF
from .frame_defs import FrameHeader as FH
from .frame_defs import FrameType as FT
from .frame_defs import _mask
from .types import FRAME_DTYPE, BasicFrameArray, FrameArrayType

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias


class FrameIllegalError(ValueError):
    """Frame is illegal."""

    pass


class ShapeError(ValueError):
    """Exception for incorrect shape."""

    pass


class TruncationWarning(UserWarning):
    """Value out of range & will be truncated."""

    pass


OUT_OF_RANGE_WARNING = "{0} out of range, will be truncated into {1} bits, {2}."


def header2type(header: FH) -> FT:
    if header <= FH.CONFIG_TYPE4:
        return FT.FRAME_CONFIG
    elif header <= FH.TEST_TYPE4:
        return FT.FRAME_TEST
    elif header <= FH.WORK_TYPE4:
        return FT.FRAME_WORK

    raise FrameIllegalError(f"unknown header: {header}.")


def header_check(frames: FrameArrayType, expected_type: FH) -> None:
    """Check the header of frame arrays.

    TODO Is it necessary to deal with the occurrence of illegal frames? Filter & return.
    """
    header0 = FH((int(frames[0]) >> FF.GENERAL_HEADER_OFFSET) & FF.GENERAL_HEADER_MASK)

    if header0 is not expected_type:
        raise ValueError(
            f"expected frame type {expected_type.name}, but got {header0.name}."
        )

    headers = (frames >> FF.GENERAL_HEADER_OFFSET) & FF.GENERAL_HEADER_MASK

    if np.unique(headers).size != 1:
        raise ValueError(
            "the header of the frame is not the same, please check the frames value."
        )


def frame_array2np(frame_array: BasicFrameArray) -> FrameArrayType:
    if isinstance(frame_array, int):
        return np.asarray([frame_array], dtype=FRAME_DTYPE)

    elif isinstance(frame_array, np.ndarray):
        if frame_array.ndim != 1:
            warnings.warn(
                f"ndim of frame arrays must be 1, but got {frame_array.ndim}. Flatten anyway.",
                UserWarning,
            )
        return frame_array.flatten().astype(FRAME_DTYPE)

    elif isinstance(frame_array, (list, tuple)):
        return np.asarray(frame_array, dtype=FRAME_DTYPE)

    else:
        raise TypeError(
            f"expected int, list, tuple or np.ndarray, but got {type(frame_array)}."
        )


def print_frame(frames: FrameArrayType) -> None:
    for frame in frames:
        print(bin(frame)[2:].zfill(64))


def np2npy(fp: Path, d: np.ndarray) -> None:
    np.save(fp, d)


def np2bin(fp: Path, d: np.ndarray) -> None:
    d.tofile(fp)


def np2txt(fp: Path, d: np.ndarray) -> None:
    with open(fp, "w") as f:
        for i in range(d.size):
            f.write("{:064b}\n".format(d[i]))


_HighBit: TypeAlias = int
_LowBit: TypeAlias = int


def bin_split(
    x: int, pos: int, high_mask_bit: Optional[int] = None
) -> tuple[_HighBit, _LowBit]:
    """Split an integer and return the high & low bits.

    Argument:
        - x: the integer.
        - pos: the position (LSB) to split the binary.
        - high_mask: mask for the high part. Optional.

    Example::

        >>> bin_split(0b1100001001, 3)
        97(0b1100001), 1
    """
    low = x & _mask(pos)

    if isinstance(high_mask_bit, int):
        high = (x >> pos) & _mask(high_mask_bit)
    else:
        high = x >> pos

    return high, low


def params_check(checker: TypeAdapter):
    def inner(func):
        @wraps(func)
        def wrapper(params: dict[str, Any], *args, **kwargs):
            checked = checker.validate_python(params)
            return func(checked, *args, **kwargs)

        return wrapper

    return inner


def params_check2(checker1: TypeAdapter, checker2: TypeAdapter):
    def inner(func):
        @wraps(func)
        def wrapper(params1: dict[str, Any], params2: dict[str, Any], *args, **kwargs):
            checked1 = checker1.validate_python(params1)
            checked2 = checker2.validate_python(params2)
            return func(checked1, checked2, *args, **kwargs)

        return wrapper

    return inner
