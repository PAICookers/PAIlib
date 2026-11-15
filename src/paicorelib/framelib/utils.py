"""Shared normalization, packing and display helpers for legacy/V2 words.

Header checks apply to standalone frames or package headers, not headerless
payloads. NumPy words are numeric values; transport serialization is separate.
"""

import warnings
from collections.abc import Iterable, Sequence
from functools import wraps
from pathlib import Path
from typing import Any, Literal, SupportsIndex, overload

import numpy as np
from numpy.typing import ArrayLike
from pydantic import TypeAdapter

from ..core_defs import LCN_EX
from ..utils import _mask
from .frame_defs import FF, FT
from .frame_defs import FrameHeader as FH
from .types import FRAME_DTYPE, FrameArrayLike, FrameArrayType


class FrameIllegalError(ValueError):
    """A frame header cannot be classified."""

    pass


class ShapeError(ValueError):
    """A frame helper received an unsupported shape."""

    pass


class TruncationWarning(UserWarning):
    """A generator discards bits or reduces precision; this is not a validation error."""

    pass


OUT_OF_RANGE_WARNING = "{0} out of range, will be truncated into {1} bits, {2}."


def header2type(header: FH) -> FT:
    if header <= FH.CONFIG_TYPE4:
        return FT.CONFIG
    elif header <= FH.TEST_TYPE4:
        return FT.TEST
    elif header <= FH.WORK_TYPE4:
        return FT.WORK
    elif header <= FH.CTRL_TYPE4:
        return FT.CTRL

    raise FrameIllegalError(f"unknown header: {header}.")


def single_frame_header_check(frame: FRAME_DTYPE, expected: FH) -> bool:
    # Both for v1 & v2
    return (int(frame) >> FF.GENERAL_HEADER_OFFSET) & FF.GENERAL_HEADER_MASK == expected


@overload
def pack_field(value: int | np.generic, offset: int, mask: int) -> int: ...


@overload
def pack_field(value: np.ndarray, offset: int, mask: int) -> np.ndarray: ...


def pack_field(
    value: int | np.ndarray | np.generic, offset: int, mask: int
) -> int | np.ndarray:
    """Mask low bits and shift; no range checks or truncation warning.

    mask is unshifted: ``pack_field(0x1f, 4, 0xf) == 0xf0``. Array shape/dtype
    arithmetic is preserved; choose a dtype wide enough for the shift.
    """
    if isinstance(value, np.ndarray):
        return (value & mask) << offset

    return (int(value) & mask) << offset


def framearray_header_check(
    frames: FrameArrayType, expected_type: FH, strict: bool = True
) -> bool:
    """Check uniform headers in a nonempty array of standalone frames.

    Mismatches raise ValueError in strict mode, otherwise return False.
    Empty input raises IndexError; do not use on whole config packages.
    """
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


def frame_array2np(frame_array: FrameArrayLike) -> FrameArrayType:
    """Normalize integer words/iterables to a new flat uint64 array.

    Scalars become (1,); generators are consumed once; multidimensional arrays
    flatten in C order with UserWarning. Strings/bytes raise TypeError. NumPy
    casts can narrow/wrap; this does not validate word ranges or decode bytes.
    """
    if isinstance(frame_array, (int, np.integer)):
        return np.asarray([frame_array], dtype=FRAME_DTYPE)

    elif isinstance(frame_array, (str, bytes, bytearray)):
        raise TypeError(
            f"expected int, iterable or np.ndarray, but got {type(frame_array).__name__}."
        )

    elif isinstance(frame_array, np.ndarray):
        if frame_array.ndim != 1:
            warnings.warn(
                f"ndim of frame arrays must be 1, but got {frame_array.ndim}. Flatten anyway.",
                UserWarning,
            )
        # Preserve input isolation without a separate flattening copy.
        return frame_array.astype(FRAME_DTYPE, order="C").reshape(-1)

    elif isinstance(frame_array, Iterable):
        return np.fromiter(frame_array, dtype=FRAME_DTYPE)

    else:
        raise TypeError(
            f"expected int, iterable or np.ndarray, but got {type(frame_array).__name__}."
        )


# Frame field widths for formatting
_FRAME_COMMON_WIDTHS = [4, 5, 5, 5, 5, 5, 5]
OFF_FRAME_GENERAL_WIDTHS = _FRAME_COMMON_WIDTHS + [30]
OFF_FRAME_WORK1_WIDTHS = _FRAME_COMMON_WIDTHS + [3, 11, 8, 8]
ON_FRAME_WORK1_1_WIDTHS = _FRAME_COMMON_WIDTHS + [3, 11, 5, 3, 8]

# For chip v2.5
LCN_TO_TS_AXON_WIDTHS = (
    (8, 9),
    (7, 10),
    (6, 11),
    (5, 12),
    (4, 13),
    (3, 14),
    (2, 15),
    (1, 16),
)
_FRAME_COMMON_WIDTHS_V2 = [4, 6, 6, 6, 6, 6, 6]
OFF_FRAME_GENERAL_WIDTHS_V2 = _FRAME_COMMON_WIDTHS_V2 + [24]
OFF_FRAME_WORK1_WIDTHS_V2 = [3, 1, 6, 6, 6, 6, 6, 6] + [7, 9, 8]


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
    widths: Sequence[int] | None = None,
    version: Literal[1, 2] = 1,
    *,
    sep: str = "_",
    reverse: bool = False,
    target_lcn: LCN_EX | None = None,
) -> list[str]:
    if version == 1:
        widths = widths or OFF_FRAME_GENERAL_WIDTHS
    else:
        if widths is None:
            if target_lcn is None:
                widths = OFF_FRAME_GENERAL_WIDTHS_V2
            else:
                TS_WIDTH, AX_WIDTH = LCN_TO_TS_AXON_WIDTHS[target_lcn.value]
                widths = OFF_FRAME_WORK1_WIDTHS_V2.copy()
                widths[-3] = TS_WIDTH - 1
                widths[-2] = AX_WIDTH
        else:
            widths = widths

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


_HighBit = int
_LowBit = int


def bin_split(
    x: int, pos: int, high_mask_bit: int | None = None
) -> tuple[_HighBit, _LowBit]:
    """Return (high, low) at bit pos; high_mask_bit optionally limits the high width."""
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
