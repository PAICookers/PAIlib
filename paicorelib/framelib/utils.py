import warnings
from collections.abc import Sequence
from functools import wraps
from pathlib import Path
from typing import Any, SupportsIndex, TypeAlias

import numpy as np
from numpy.typing import ArrayLike
from pydantic import TypeAdapter

from ..utils import _mask
from .frame_defs import FrameFormat as FF
from .frame_defs import FrameHeader as FH
from .frame_defs import FrameType as FT
from .types import FRAME_DTYPE, BasicFrameArray, FrameArrayType


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
        return FT.CONFIG
    elif header <= FH.TEST_TYPE4:
        return FT.TEST
    elif header <= FH.WORK_TYPE4:
        return FT.WORK

    raise FrameIllegalError(f"unknown header: {header}.")


def framearray_header_check(
    frames: FrameArrayType, expected_type: FH, strict: bool = True
) -> bool:
    """Check the header of frame arrays."""
    header0 = FH((int(frames[0]) >> FF.GENERAL_HEADER_OFFSET) & FF.GENERAL_HEADER_MASK)

    if header0 != expected_type:
        if strict:
            raise ValueError(
                f"expected frame type {expected_type.name}, but got {header0.name}."
            )
        else:
            return False

    headers = (frames >> FF.GENERAL_HEADER_OFFSET) & FF.GENERAL_HEADER_MASK

    if np.unique(headers).size > 1:
        if strict:
            raise ValueError(
                "the headers of the frame are not all the same, please check the frames value."
            )
        else:
            return False

    return True


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
            f"expected int, list, tuple or np.ndarray, but got {type(frame_array).__name__}."
        )


# Frame field widths for formatting
_FRAME_COMMON_WIDTHS = [4, 5, 5, 5, 5, 5, 5]
OFF_FRAME_GENERAL_WIDTHS = _FRAME_COMMON_WIDTHS + [30]
OFF_FRAME_WORK1_WIDTHS = _FRAME_COMMON_WIDTHS + [3, 11, 8, 8]
ON_FRAME_WORK1_1_WIDTHS = _FRAME_COMMON_WIDTHS + [3, 11, 5, 3, 8]


def format_frame_bin(
    value: SupportsIndex,
    widths: Sequence[int] = OFF_FRAME_GENERAL_WIDTHS,
    sep: str = "_",
    reverse: bool = False,
) -> str:
    total_bits = FF.FRAME_LENGTH
    bin_str = np.binary_repr(value, width=total_bits)
    parts = []
    start = 0

    for w in reversed(widths) if reverse else widths:
        parts.append(bin_str[start : start + w])
        start += w

    return sep.join(parts)


def print_frame(
    frames: ArrayLike,
    widths: Sequence[int] = OFF_FRAME_GENERAL_WIDTHS,
    *,
    sep: str = "_",
    reverse: bool = False,
) -> list[str]:
    s = [
        format_frame_bin(f, widths, sep, reverse)
        for f in np.asarray(frames, FRAME_DTYPE).flat
    ]
    for line in s:
        print(line)

    return s


def np2npy(fp: Path, d: np.ndarray) -> None:
    assert fp.suffix == ".npy"
    np.save(fp, d)


def np2bin(fp: Path, d: np.ndarray) -> None:
    assert fp.suffix == ".bin"
    d.tofile(fp)


def np2txt(fp: Path, d: np.ndarray) -> None:
    assert fp.suffix == ".txt"

    with fp.open("w") as f:
        for i in range(d.size):
            f.write(f"{d[i]:0{FF.FRAME_LENGTH}b}\n")


_HighBit: TypeAlias = int
_LowBit: TypeAlias = int


def bin_split(
    x: int, pos: int, high_mask_bit: int | None = None
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
            validated = checker.validate_python(params).model_dump()  # return dict
            return func(validated, *args, **kwargs)

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
