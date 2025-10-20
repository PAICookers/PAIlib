import numpy as np
import pytest

from paicorelib.framelib.frame_defs import FrameHeader as FH
from paicorelib.framelib.types import FRAME_DTYPE
from paicorelib.framelib.utils import (
    framearray_header_check,
    print_frame,
    OFF_FRAME_WORK1_WIDTHS,
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
