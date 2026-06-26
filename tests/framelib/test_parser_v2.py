import numpy as np
import pytest

from paicorelib.coordinate import CoordXY, CoordZXYOffset, coordzxy_to_sign_magnitude
from paicorelib.core_defs import LCN_EX
from paicorelib.core_defs_v2 import DataWidth
from paicorelib.framelib.frame_defs import FFV2
from paicorelib.framelib.frame_defs import FrameHeader as FH
from paicorelib.framelib.frame_defs import OfflineConfigFrame1FormatV2 as Off_Cfg1_V2
from paicorelib.framelib.frame_gen_v2 import OfflineFrameGenV2
from paicorelib.framelib.parser_v2 import (
    FrameParseError,
    bit_field,
    decode_aer_route_fields,
    decode_core_config,
    decode_header,
    decode_lut_entries,
    expand_aer_destinations,
    parse_frame_stream,
    sign_magnitude_to_int,
)
from paicorelib.routing_hexa import AERPacket, AERPacketZXYCopy, aer_packet_walk
from tests.utils import build_v2_core_reg_params


def _cf1(coord: CoordZXYOffset, **overrides) -> np.ndarray:
    return OfflineFrameGenV2.gen_config_frame1(
        coord, build_v2_core_reg_params(**overrides)
    )


def _cf2(coord: CoordZXYOffset) -> np.ndarray:
    potential = np.arange(-128, 128, dtype=np.int32)
    activation = np.arange(256, dtype=np.uint8)
    return OfflineFrameGenV2.gen_config_frame2(coord, potential, activation)


def _cf3_header(coord: CoordZXYOffset, start_addr: int, n_package: int) -> np.ndarray:
    return OfflineFrameGenV2.gen_config_frame3_pkg_header(coord, start_addr, n_package)


def test_decode_header_and_type1_core_config() -> None:
    frames = _cf1(
        CoordZXYOffset(1, 2, -1),
        neuron_number=123,
        lcn=LCN_EX.LCN_4X,
        target_lcn=LCN_EX.LCN_8X,
        input_width=DataWidth.WIDTH_4BIT,
        test_core_xy=-3,
        test_core_x=1,
        test_core_y=0,
        tick_start=7,
    )

    header = decode_header(int(frames[0]))
    config = decode_core_config(tuple(int(v) for v in frames[1:]))

    assert header.header == FH.CONFIG_TYPE1
    assert header.offset == CoordZXYOffset(1, 2, -1)
    assert header.coord_key == (3, 0)
    assert config["neuron_number"] == 123
    assert config["lcn"] == LCN_EX.LCN_4X.value
    assert config["target_lcn"] == LCN_EX.LCN_8X.value
    assert config["input_width"] == DataWidth.WIDTH_4BIT.value
    assert config["tick_start"] == 7
    assert sign_magnitude_to_int(config["test_core_xy"]) == -3
    assert sign_magnitude_to_int(config["test_core_x"]) == 1
    assert sign_magnitude_to_int(config["test_core_y"]) == 0


def test_sign_magnitude_all_ones_is_negative_31() -> None:
    assert sign_magnitude_to_int(0b111111) == -31


def test_parse_frame_stream_accepts_ordered_config_packages() -> None:
    frames = np.concatenate(
        [
            _cf1(CoordZXYOffset(0, 1, 1)),
            _cf2(CoordZXYOffset(0, 1, 1)),
            _cf3_header(CoordZXYOffset(0, 1, 1), start_addr=12, n_package=2),
            np.array([0x11, 0x22], dtype=np.uint64),
        ]
    )
    stream = parse_frame_stream(int(value) for value in frames)
    core = stream.cores[(1, 1)]

    assert [package.frame_type for package in stream.packages] == [1, 2, 3]
    assert [package.frame_type for package in core.packages] == [1, 2, 3]
    assert len(core.frame_type1_payloads) == 3
    assert len(core.frame_type2_payloads) == 256
    assert len(core.frame_type3_payloads) == 2


def test_parse_frame_stream_accepts_repeated_and_unordered_config_packages() -> None:
    stream = parse_frame_stream(
        np.concatenate(
            [
                _cf1(CoordZXYOffset(0, 1, 1), neuron_number=1),
                _cf1(CoordZXYOffset(0, 1, 1), neuron_number=2),
                _cf3_header(CoordZXYOffset(0, 1, 1), start_addr=0, n_package=2),
                np.array([0x33, 0x44], dtype=np.uint64),
                _cf2(CoordZXYOffset(0, 1, 1)),
            ]
        )
    )
    core = stream.cores[(1, 1)]
    config = decode_core_config(core.frame_type1_payloads)

    assert [package.frame_type for package in core.packages] == [1, 1, 3, 2]
    assert len(core.frame_type1_payloads) == 6
    assert config["neuron_number"] == 1


def test_parse_frame_stream_groups_interleaved_cores() -> None:
    stream = parse_frame_stream(
        np.concatenate(
            [
                _cf1(CoordZXYOffset(0, 1, 1), neuron_number=11),
                _cf1(CoordZXYOffset(0, 2, 2), neuron_number=22),
                _cf3_header(CoordZXYOffset(0, 1, 1), start_addr=4, n_package=2),
                np.array([0x55, 0x66], dtype=np.uint64),
            ]
        )
    )

    assert set(stream.cores) == {(1, 1), (2, 2)}
    assert [package.frame_type for package in stream.packages] == [1, 1, 3]
    assert [package.frame_type for package in stream.cores[(1, 1)].packages] == [1, 3]
    assert [package.frame_type for package in stream.cores[(2, 2)].packages] == [1]


def test_parse_frame_stream_reports_truncated_package() -> None:
    frames = _cf3_header(CoordZXYOffset(0, 1, 1), start_addr=0, n_package=2)

    with pytest.raises(FrameParseError, match="truncated") as exc_info:
        parse_frame_stream(frames)

    assert exc_info.value.frame_index == 0
    assert exc_info.value.raw_frame == int(frames[0])
    assert exc_info.value.context["package_count"] == 2


def test_parse_frame_stream_reports_unsupported_header() -> None:
    bad = np.array([int(FH.WORK_TYPE1) << FFV2.GENERAL_HEADER_OFFSET], dtype=np.uint64)

    with pytest.raises(FrameParseError, match="unsupported") as exc_info:
        parse_frame_stream(bad)

    assert exc_info.value.frame_index == 0
    assert exc_info.value.context["header"] == int(FH.WORK_TYPE1)


def test_decode_lut_entries_matches_generated_type2_payload() -> None:
    frames = _cf2(CoordZXYOffset(0, 1, 1))
    entries = decode_lut_entries(tuple(int(v) for v in frames[1:]))

    assert len(entries) == 256
    assert entries[0].potential == -128
    assert entries[0].activation == 0
    assert entries[-1].potential == 127
    assert entries[-1].activation == 255


def test_expand_aer_destinations_matches_paicorelib_walk() -> None:
    offset = CoordZXYOffset(0, 3, 2)
    ncopy = AERPacketZXYCopy(0, 1, 1)
    raw_offset = coordzxy_to_sign_magnitude(offset)
    raw_copy = ncopy.to_sign_magnitude()
    route = decode_aer_route_fields(
        addr_core_xy=raw_offset[0],
        addr_core_x=raw_offset[1],
        addr_core_y=raw_offset[2],
        addr_copy_xy=raw_copy[0],
        addr_copy_x=raw_copy[1],
        addr_copy_y=raw_copy[2],
    )
    source = CoordXY(1, 1)

    assert expand_aer_destinations(source, route) == aer_packet_walk(
        AERPacket(source.copy(), offset, ncopy)
    )


def test_decode_core_config_uses_first_complete_type1_group() -> None:
    first = _cf1(CoordZXYOffset(0, 1, 1), neuron_number=10)
    second = _cf1(CoordZXYOffset(0, 1, 1), neuron_number=20)

    config = decode_core_config(
        tuple(int(v) for v in np.concatenate([first[1:], second[1:]]))
    )

    assert config["neuron_number"] == 10


def test_decode_core_config_reconstructs_split_test_core_y_field() -> None:
    frames = _cf1(CoordZXYOffset(0, 1, 1), test_core_y=-31)
    w1, w2 = int(frames[1]), int(frames[2])
    expected_raw_y = coordzxy_to_sign_magnitude(CoordZXYOffset(0, 0, -31))[2]

    config = decode_core_config(tuple(int(v) for v in frames[1:]))

    assert config["test_core_y"] == expected_raw_y
    assert (
        bit_field(
            w1,
            Off_Cfg1_V2.Word1.TEST_CORE_Y_HIGH2_OFFSET,
            Off_Cfg1_V2.Word1.TEST_CORE_Y_HIGH2_MASK,
        )
        == 0b11
    )
    assert (
        bit_field(
            w2,
            Off_Cfg1_V2.Word2.TEST_CORE_Y_LOW4_OFFSET,
            Off_Cfg1_V2.Word2.TEST_CORE_Y_LOW4_MASK,
        )
        == 0b1111
    )
