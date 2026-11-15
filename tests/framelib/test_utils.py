import numpy as np
import pytest

from paicorelib.framelib.frame_defs import FrameHeader as FH
from paicorelib.framelib.types import FRAME_DTYPE
from paicorelib.framelib.utils import (
    OFF_FRAME_WORK1_WIDTHS,
    frame_array2np,
    framearray_header_check,
    print_frame,
)


@pytest.mark.parametrize(
    "frames",
    [
        # wrong header
        np.array(
            [
                0b0100_0000100001_00011_11100_00000_00001_000_00000000000_00000000_01010101,
                0b0100_0000100001_00011_11100_00000_00001_000_00000000000_00000000_00000001,
                0b0100_0000100001_00011_11100_00000_00001_000_00000000000_00000000_01000000,
                0b0100_0000100001_00011_11100_00000_00001_000_00000000000_00000000_01000000,
                0b0101_0000100001_00011_11100_00000_00001_000_00000000000_00000000_01000000,
                0b0110_0000100001_00011_11100_00000_00001_000_00000000000_00000000_01000000,
            ],
            dtype=FRAME_DTYPE,
        )
    ],
)
def test_framearray_header_check(frames):
    with pytest.raises(ValueError):
        framearray_header_check(frames, FH.WORK_TYPE1, strict=True)


@pytest.mark.parametrize(
    "frames",
    [
        np.array(
            [
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000000_00000001_00000001,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000011_00000000_00000111,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000101_00000000_00001000,
            ],
            dtype=FRAME_DTYPE,
        ),
        0b1000_00001_00000_00000_00000_00000_00000_000_00000000000_00000001_00000001,
    ],
)
def test_print_frame(frames):
    print_frame(frames, OFF_FRAME_WORK1_WIDTHS)


@pytest.mark.parametrize(
    ("frames", "expected"),
    [(np.uint64(7), [7]), ((value for value in [1, 2]), [1, 2]), (range(3), [0, 1, 2])],
)
def test_frame_array2np_accepts_scalars_and_iterables(frames, expected):
    array = frame_array2np(frames)

    assert array.dtype == FRAME_DTYPE
    assert array.tolist() == expected


@pytest.mark.parametrize(
    ("frames", "expected"),
    [
        (np.array([0, 2**63, 2**64 - 1], dtype=np.uint64), [0, 2**63, 2**64 - 1]),
        (np.arange(6, dtype=np.uint64)[::-2], [5, 3, 1]),
        (np.array([-1, 0, 7], dtype=np.int16), [2**64 - 1, 0, 7]),
        (np.array([0, 2**64 - 1], dtype=">u8"), [0, 2**64 - 1]),
        (np.array([], dtype=np.uint64), []),
        (np.broadcast_to(np.uint64(5), (3,)), [5, 5, 5]),
    ],
)
def test_frame_array2np_returns_independent_writable_words(frames, expected):
    frames.setflags(write=False)
    array = frame_array2np(frames)

    assert array.dtype == FRAME_DTYPE
    assert array.ndim == 1
    assert array.flags.c_contiguous
    assert array.flags.writeable
    assert array.tolist() == expected
    assert not np.shares_memory(array, frames)
    if array.size:
        original_first = int(frames[0])
        array[0] = 123
        assert int(frames[0]) == original_first


@pytest.mark.parametrize(
    ("frames", "expected"),
    [
        (np.array(7, dtype=np.uint64), [7]),
        (np.arange(6, dtype=np.uint64).reshape(2, 3).T, [0, 3, 1, 4, 2, 5]),
        (np.asfortranarray([[0, 1], [2, 3]]), [0, 1, 2, 3]),
        (np.empty((2, 0), dtype=np.uint64), []),
    ],
)
def test_frame_array2np_flattens_in_c_order_with_warning(frames, expected):
    with pytest.warns(UserWarning, match="Flatten anyway"):
        array = frame_array2np(frames)

    assert array.dtype == FRAME_DTYPE
    assert array.ndim == 1
    assert array.flags.c_contiguous
    assert array.tolist() == expected
    assert not np.shares_memory(array, frames)
