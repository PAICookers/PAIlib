import contextlib
import math
from typing import Any, Literal, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from paicorelib.coordinate import CoordZXYOffset, coordzxy_to_sign_magnitude
from paicorelib.core_defs import LCN_EX
from paicorelib.core_defs_v2 import CSCAccelerateMode, DataSign, DataWidth
from paicorelib.core_model_v2 import OfflineCoreRegV2, OnlineCoreRegV2
from paicorelib.float_codec import (
    pack_bf16_payload_bits,
    pack_bf16_scalar_bits,
    pack_fp32_scalar_bits,
)
from paicorelib.framelib import FRAME_DTYPE
from paicorelib.framelib.base import get_frame_dest_v2
from paicorelib.framelib.frame_defs import FFV2, FramePackageType
from paicorelib.framelib.frame_defs import FrameHeader as FH
from paicorelib.framelib.frame_defs import (
    OfflineConfigFrame1FormatV2 as Off_Cfg1_V2,
)
from paicorelib.framelib.frame_defs import (
    OfflineConfigFrame2FormatV2 as Off_Cfg2_V2,
)
from paicorelib.framelib.frame_defs import (
    OfflineConfigFrame3FormatV2 as Off_Cfg3_V2,
)
from paicorelib.framelib.frame_defs import (
    OfflineWorkFrame1FormatV2 as Off_Work1_V2,
)
from paicorelib.framelib.frame_defs import (
    OfflineWorkFrame2FormatV2 as Off_Work2_V2,
)
from paicorelib.framelib.frame_defs import OnlineConfigFrame1FormatV2 as On_Cfg1_V2
from paicorelib.framelib.frame_defs import OnlineConfigFrame2FormatV2 as On_Cfg2_V2
from paicorelib.framelib.frame_defs import OnlineConfigFrame3FormatV2 as On_Cfg3_V2
from paicorelib.framelib.frame_defs import OnlineControlFrame4FormatV2 as On_Ctrl4_V2
from paicorelib.framelib.frame_defs import OnlineWorkFrame1FormatV2 as On_Work1_V2
from paicorelib.framelib.frame_defs import OnlineWorkFrame2FormatV2 as On_Work2_V2
from paicorelib.framelib.frame_defs import OnlineWorkFrame3FormatV2 as On_Work3_V2
from paicorelib.framelib.frame_defs import OnlineWorkFrame4FormatV2 as On_Work4_V2
from paicorelib.framelib.frame_gen_v2 import (
    DataWidthLE8Like,
    FrameGenV2,
    OfflineFrameGenV2,
    OnlineFrameGenV2,
    online_weight_csc_pack,
    online_weight_dense_pack,
    weight_csc_pack,
    weight_dense_pack,
)
from paicorelib.framelib.types import (
    LUT_ACTIVATION_DTYPE,
    LUT_POTENTIAL_DTYPE,
    FrameArrayType,
)
from paicorelib.framelib.utils import TruncationWarning, single_frame_header_check
from paicorelib.neuron_defs import ResetMode
from paicorelib.neuron_defs_v2 import (
    FoldType,
    LateralInhibitionMode,
    LeakAddMode,
    LeakMultiComparisonOrder,
    LeakMultiInputMode,
    LeakMultiMode,
    NeuronType,
    OnlineOutputType,
    OutputType,
    ThresholdNegMode,
    ThresholdPosMode,
    WeightCompressType,
)
from paicorelib.neuron_model_v2 import (
    OfflineNeuDestInfoV2,
    OfflineNeuFoldedAttrsV2Part1,
    OfflineNeuFoldedAttrsV2Part2,
    OfflineNeuFullAttrsV2Part1,
    OfflineNeuFullAttrsV2Part2,
    OnlineNeuDestInfoV2,
    OnlineNeuFoldedAttrsV2Part1,
    OnlineNeuFoldedAttrsV2Part2,
    OnlineNeuFullAttrsV2Part2,
    OnlineNeuHalfAttrsV2,
)
from paicorelib.routing_hexa import AERPacketZXYCopy
from paicorelib.utils import _mask
from tests.utils import (
    bit_field,
    build_online_v2_core_reg_params,
    build_online_v2_half_attrs_params,
    build_v2_core_reg_params,
    build_v2_dest_info_params,
    build_v2_folded_attrs_part1_params,
    build_v2_folded_attrs_part2_params,
    build_v2_full_attrs_part2_params,
    build_v2_half_attrs_params,
    gen_random_array,
)

WidthBitsLE8 = Literal[1, 2, 4, 8]


def parse_package_header(frames: FrameArrayType) -> tuple[int, int, int]:
    payload = bit_field(
        frames[0], FFV2.GENERAL_PAYLOAD_OFFSET, FFV2.GENERAL_PAYLOAD_MASK
    )
    pkg_type = (
        payload >> FFV2.GENERAL_PACKAGE_TYPE_OFFSET
    ) & FFV2.GENERAL_PACKAGE_TYPE_MASK
    start_addr = (
        payload >> FFV2.GENERAL_PACKAGE_NEU_START_ADDR_OFFSET
    ) & FFV2.GENERAL_PACKAGE_NEU_START_ADDR_MASK
    n_package = (
        payload >> FFV2.GENERAL_PACKAGE_NUM_OFFSET
    ) & FFV2.GENERAL_PACKAGE_NUM_MASK
    return pkg_type, start_addr, n_package


def extract_lut_from_cf2(cf2: FrameArrayType, act_dtype: LUT_ACTIVATION_DTYPE):
    assert cf2.size == 257
    assert single_frame_header_check(cf2[0], FH.CONFIG_TYPE2)

    lut = cf2[1:]
    arr_pot = (
        (lut >> Off_Cfg2_V2.POTENTIAL_OFFSET) & Off_Cfg2_V2.POTENTIAL_MASK
    ).astype(LUT_POTENTIAL_DTYPE)
    arr_act = (
        (lut >> Off_Cfg2_V2.ACTIVATION_OFFSET) & Off_Cfg2_V2.ACTIVATION_MASK
    ).astype(act_dtype)
    return arr_pot, arr_act


def extract_online_lut_from_cf2(cf2: FrameArrayType):
    assert cf2.size == 257
    assert single_frame_header_check(cf2[0], FH.CONFIG_TYPE2)

    lut = cf2[1:]
    arr_pot = (
        ((lut >> On_Cfg2_V2.POTENTIAL_OFFSET) & On_Cfg2_V2.POTENTIAL_MASK)
        .astype(np.uint32)
        .view(np.int32)
    )
    arr_act_bits = (
        (lut >> On_Cfg2_V2.ACTIVATION_OFFSET) & On_Cfg2_V2.ACTIVATION_MASK
    ).astype(np.uint16)
    return arr_pot, arr_act_bits


def normalize_width_bits(value: DataWidthLE8Like) -> WidthBitsLE8:
    width_bits = (1 << value.value) if isinstance(value, DataWidth) else int(value)
    return cast(WidthBitsLE8, width_bits)


def gen_v2_weight_array(
    size: int,
    weight_width: int,
    signed: bool,
    rng: np.random.Generator | None = None,
    sparse_ratio: float = 0.0,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    if signed:
        dtype = np.int8
        max_value = _mask(weight_width - 1)
        min_value = -(max_value + 1)
    else:
        dtype = np.uint8
        min_value, max_value = 0, _mask(weight_width)

    weight = rng.integers(min_value, max_value, size=size, dtype=dtype, endpoint=True)

    if sparse_ratio > 0 and weight.size > 0:
        n_zero = int(weight.size * sparse_ratio)
        if n_zero > 0:
            indices = rng.choice(weight.size, size=n_zero, replace=False)
            weight.flat[indices] = 0

    return weight


def validated_dump(model_cls, params: dict[str, Any]) -> dict[str, Any]:
    return model_cls.model_validate(params, strict=True).model_dump()


def _target_lcn_index(target_lcn: LCN_EX | int) -> int:
    return target_lcn.value if isinstance(target_lcn, LCN_EX) else target_lcn


def _offline_work_ts_ax_addr(
    timesteps, axons, target_lcn: LCN_EX | int, F: type[Off_Work1_V2 | Off_Work2_V2]
) -> FrameArrayType:
    ts_width, ax_width = OfflineFrameGenV2.LCN_TO_TS_AXON_WIDTHS[
        _target_lcn_index(target_lcn)
    ]
    ts = np.asarray(timesteps, dtype=FRAME_DTYPE).ravel()
    ax = np.asarray(axons, dtype=FRAME_DTYPE).ravel()
    ts_msb = (ts >> (ts_width - 1)) & F.TIMESTEP_HIGH7_MASK
    ts_low = ts & _mask(ts_width - 1)
    return (
        (ts_msb << F.TIMESTEP_HIGH7_OFFSET)
        | (ts_low << (F.AXON_ADDR_OFFSET + ax_width))
        | ((ax & _mask(ax_width)) << F.AXON_ADDR_OFFSET)
    ).astype(FRAME_DTYPE, copy=False)


def _online_work_ts_ax_addr(
    timesteps,
    axons,
    target_lcn: LCN_EX | int,
    F: type[On_Work1_V2 | On_Work2_V2 | On_Work3_V2 | On_Work4_V2],
) -> FrameArrayType:
    ts_width, ax_width = OnlineFrameGenV2.LCN_TO_TS_AXON_WIDTHS[
        _target_lcn_index(target_lcn)
    ]
    ts = np.asarray(timesteps, dtype=FRAME_DTYPE).ravel()
    ax = np.asarray(axons, dtype=FRAME_DTYPE).ravel()
    return (
        (
            (((ts & _mask(ts_width)) << ax_width) | (ax & _mask(ax_width)))
            & F.TIMESTEP_AXON_MASK
        )
        << F.TIMESTEP_AXON_OFFSET
    ).astype(FRAME_DTYPE, copy=False)


def _payload_le_byte_rows(payload: np.ndarray) -> NDArray[np.uint8]:
    payload = np.ascontiguousarray(
        payload.astype(payload.dtype.newbyteorder("<"), copy=False)
    )
    return payload.view(np.uint8).reshape(payload.size, payload.dtype.itemsize)


def _expected_online_work_frames(
    header: FH,
    F: type[On_Work1_V2 | On_Work2_V2 | On_Work3_V2 | On_Work4_V2],
    pkt_offset: CoordZXYOffset,
    pkt_ncopy: AERPacketZXYCopy,
    timesteps: np.ndarray,
    axons: np.ndarray,
    target_lcn: LCN_EX | int,
    payload: np.ndarray,
) -> tuple[FrameArrayType, np.ndarray, np.ndarray]:
    payload = np.asarray(payload).ravel()
    mask = np.flatnonzero(payload)
    if mask.size == 0:
        return (
            np.array([], dtype=FRAME_DTYPE),
            np.array([], dtype=np.uint8),
            np.array([], dtype=FRAME_DTYPE),
        )

    payload_parts = _payload_le_byte_rows(payload[mask])
    expected_bytes = payload_parts.ravel()
    expected_ts_ax_addr = np.repeat(
        _online_work_ts_ax_addr(timesteps, axons, target_lcn, F)[mask],
        payload_parts.shape[1],
    )
    expected_frames = (
        get_frame_dest_v2(header, pkt_offset, pkt_ncopy)
        + expected_ts_ax_addr
        + expected_bytes
    ).astype(FRAME_DTYPE)
    return expected_frames, expected_bytes, expected_ts_ax_addr


def _make_online_float_payload(
    n: int, dtype: type[np.float16] | type[np.float32], rng: np.random.Generator
) -> np.ndarray:
    payload = rng.uniform(-16.0, 16.0, n).astype(dtype)
    payload[::17] = 0
    payload[1::43] = dtype(-0.0)
    payload[2] = dtype(1.5)
    payload[3] = dtype(-2.5)
    return payload


def _online_work_batch_cases():
    rng = np.random.default_rng()
    n = 384

    wf1_bool = gen_random_array((n,), np.bool_, rng, sparse_ratio=1 / 4)
    wf1_bool[:4] = np.array([False, True, False, True])

    wf1_u1 = rng.integers(0, 1, size=n, dtype=np.uint8, endpoint=True)
    wf1_u1[rng.choice(n, size=n // 4, replace=False)] = 0
    wf1_u1[:4] = np.array([0, 1, 0, 1], dtype=np.uint8)

    return [
        ("gen_work_frame1", FH.WORK_TYPE1, On_Work1_V2, LCN_EX.LCN_8X, wf1_bool),
        ("gen_work_frame1", FH.WORK_TYPE1, On_Work1_V2, LCN_EX.LCN_8X.value, wf1_u1),
        (
            "gen_work_frame1",
            FH.WORK_TYPE1,
            On_Work1_V2,
            LCN_EX.LCN_8X,
            _make_online_float_payload(n, np.float16, rng),
        ),
        (
            "gen_work_frame2",
            FH.WORK_TYPE2,
            On_Work2_V2,
            LCN_EX.LCN_8X,
            _make_online_float_payload(n, np.float16, rng),
        ),
        (
            "gen_work_frame3",
            FH.WORK_TYPE3,
            On_Work3_V2,
            LCN_EX.LCN_8X,
            _make_online_float_payload(n, np.float32, rng),
        ),
        (
            "gen_work_frame3",
            FH.WORK_TYPE3,
            On_Work3_V2,
            LCN_EX.LCN_8X.value,
            _make_online_float_payload(n, np.float16, rng),
        ),
        (
            "gen_work_frame4",
            FH.WORK_TYPE4,
            On_Work4_V2,
            LCN_EX.LCN_8X,
            _make_online_float_payload(n, np.float16, rng),
        ),
    ]


def _unpack_unsigned_groups(
    frames: np.ndarray, bit_width: int, group_size: int, *, dtype
) -> np.ndarray:
    frames_arr = np.asarray(frames, dtype=FRAME_DTYPE)
    shifts = bit_width * np.arange(group_size, dtype=np.uint8)
    mask = np.asarray(_mask(bit_width), dtype=FRAME_DTYPE)
    groups = (frames_arr[:, np.newaxis] >> shifts) & mask
    return groups.astype(dtype, copy=False)


def _restore_signed_weights(
    values: np.ndarray, weight_width: int, signed: bool
) -> np.ndarray:
    if not signed or weight_width >= 8:
        return values.astype(np.int8 if signed else np.uint8, copy=False)

    signbit = 1 << (weight_width - 1)
    restored = values.astype(np.int8, copy=False)
    restored[restored >= signbit] -= 1 << weight_width
    return restored


def weight_dense_unpack(
    frames: FrameArrayType, weight_width: int, signed: bool, original_size: int
) -> NDArray[np.int8 | np.uint8]:
    weights = _unpack_unsigned_groups(
        frames, weight_width, 64 // weight_width, dtype=np.uint8
    ).ravel()[:original_size]
    return _restore_signed_weights(weights, weight_width, signed)


def weight_csc_unpack(
    frames: FrameArrayType, weight_width: int, signed: bool, original_size: int
) -> NDArray[np.int8 | np.uint8]:
    n_nonzero_w_per_addr = {1: 7, 2: 7, 4: 6, 8: 5}[weight_width]
    indices_addr_offset = {1: 16, 2: 16, 4: 32, 8: 48}[weight_width]

    if frames.size % 2 != 0:
        raise ValueError(
            f"'frames' length must be even for CSC unpack, but got {frames.size}."
        )

    chunk_low64 = frames[0::2]
    chunk_high64 = frames[1::2]
    weights = _unpack_unsigned_groups(
        chunk_low64, weight_width, n_nonzero_w_per_addr, dtype=np.uint8
    )

    n_idx_at_high = 4
    n_idx_at_low = n_nonzero_w_per_addr - n_idx_at_high
    indices_l = _unpack_unsigned_groups(
        chunk_low64 >> indices_addr_offset, 16, n_idx_at_low, dtype=np.uint16
    )
    indices_h = _unpack_unsigned_groups(
        chunk_high64, 16, n_idx_at_high, dtype=np.uint16
    )
    indices = np.hstack([indices_l, indices_h])
    weights = _restore_signed_weights(weights, weight_width, signed)

    result = np.zeros(original_size, dtype=weights.dtype)
    result[indices] = weights
    return result


def online_weight_dense_unpack(
    frames: FrameArrayType, original_size: int
) -> NDArray[np.uint16]:
    return np.ascontiguousarray(frames).view(np.uint16)[:original_size]


def online_weight_csc_unpack(
    frames: FrameArrayType, original_size: int
) -> NDArray[np.uint16]:
    if frames.size % 2 != 0:
        raise ValueError(
            f"'frames' length must be even for CSC unpack, but got {frames.size}."
        )

    weight_chunks = np.ascontiguousarray(frames[0::2]).view(np.uint16)
    index_chunks = np.ascontiguousarray(frames[1::2]).view(np.uint16)
    weights = weight_chunks.reshape(-1, 4)
    indices = index_chunks.reshape(-1, 4)

    result = np.zeros(original_size, dtype=np.uint16)
    result[indices] = weights
    return result


class TestFrameGenV2:
    def test_make_package(self, v2_packet_route):
        pkt_offset, pkt_ncopy = v2_packet_route
        packages = np.array([11, 22, 33], dtype=FRAME_DTYPE)

        frames = FrameGenV2.make_package(
            FH.CONFIG_TYPE2, pkt_offset, pkt_ncopy, 7, packages
        )

        assert frames.dtype == FRAME_DTYPE
        assert frames.shape == (4,)
        assert single_frame_header_check(frames[0], FH.CONFIG_TYPE2)
        assert parse_package_header(frames) == (
            FramePackageType.CONF_TESTOUT.value,
            7,
            3,
        )
        assert np.array_equal(frames[1:], packages)


class TestOfflineFrameGenV2:
    def test_cf1(self, v2_packet_route):
        pkt_offset, pkt_ncopy = v2_packet_route
        core_reg = OfflineCoreRegV2.model_validate(
            build_v2_core_reg_params(
                name="core_reg",
                input_width=DataWidth.WIDTH_8BIT,
                output_width=DataWidth.WIDTH_8BIT,
                weight_sign=DataSign.SIGNED,
                weight_width=DataWidth.WIDTH_8BIT,
                lcn=LCN_EX.LCN_8X,
                target_lcn=LCN_EX.LCN_8X,
                neuron_number=200,
                test_core_xy=5,
                global_send=0b100_0000,
                csc_accelerate=CSCAccelerateMode.ENABLE,
                global_receive=0b001_0000,
                busy_cycle=2,
                tick_start=1,
            ),
            strict=True,
        )

        frames = OfflineFrameGenV2.gen_config_frame1(pkt_offset, core_reg, pkt_ncopy, 0)

        assert frames.dtype == FRAME_DTYPE
        assert frames.size == 4
        assert parse_package_header(frames) == (
            FramePackageType.CONF_TESTOUT.value,
            0,
            3,
        )
        assert (
            bit_field(
                frames[1],
                Off_Cfg1_V2.Word1.TARGET_LCN_OFFSET,
                Off_Cfg1_V2.Word1.TARGET_LCN_MASK,
            )
            == LCN_EX.LCN_8X.value
        )
        assert (
            bit_field(
                frames[2],
                Off_Cfg1_V2.Word2.CSC_ACCELERATE_OFFSET,
                Off_Cfg1_V2.Word2.CSC_ACCELERATE_MASK,
            )
            == CSCAccelerateMode.ENABLE.value
        )
        assert (
            bit_field(
                frames[3],
                Off_Cfg1_V2.Word3.TICK_START_OFFSET,
                Off_Cfg1_V2.Word3.TICK_START_MASK,
            )
            == 1
        )

    @pytest.mark.parametrize("use_random_lut", [True, False])
    @pytest.mark.parametrize("act_dtype", [np.uint8, np.int8])
    def test_cf2(self, v2_packet_route, use_random_lut, act_dtype):
        pkt_offset, _ = v2_packet_route
        if use_random_lut:
            arr_pot = gen_random_array((256,), dtype=LUT_POTENTIAL_DTYPE)
            arr_act = gen_random_array((256,), dtype=act_dtype)
        else:
            arr_pot = np.arange(-128, 128, dtype=LUT_POTENTIAL_DTYPE)
            arr_act = (
                np.ones((256,), dtype=act_dtype)
                if act_dtype == np.uint8
                else np.full((256,), -1, dtype=act_dtype)
            )

        frames = OfflineFrameGenV2.gen_config_frame2(pkt_offset, arr_pot, arr_act)  # type: ignore[arg-type]

        assert frames.size == 257

        arr_pot2, arr_act2 = extract_lut_from_cf2(frames, act_dtype)
        assert np.array_equal(arr_pot, arr_pot2)
        assert np.array_equal(arr_act, arr_act2)

    @pytest.mark.parametrize("n_package", [0, 1, 100, 1000, 8000])
    def test_gen_config_frame3_pkg_header(self, v2_packet_route, n_package):
        pkt_offset, _ = v2_packet_route
        frames = OfflineFrameGenV2.gen_config_frame3_pkg_header(
            pkt_offset, 12, n_package
        )

        assert frames.dtype == FRAME_DTYPE
        assert frames.shape == (1,)
        assert single_frame_header_check(frames[0], FH.CONFIG_TYPE3)
        assert parse_package_header(frames) == (
            FramePackageType.CONF_TESTOUT.value,
            12,
            n_package,
        )

    def test_gen_pkg_half_neu_fields(self):
        dest_info = build_v2_dest_info_params(
            tick_relative=7,
            addr_axon=0x1AB,
            addr_core_xy=-1,
            addr_core_x=2,
            addr_core_y=-3,
            addr_copy_xy=4,
            addr_copy_x=5,
            addr_copy_y=-6,
        )
        half_attrs = build_v2_half_attrs_params(
            weight_skew=0x6A5,
            weight_address_start=0x123,
            weight_address_end=0x456,
            output_type=OutputType.POTENTIAL,
            fold_type=FoldType.FOLDED,
            neuron_type=NeuronType.FULL,
            vjt=0x12345678,
        )

        dest_info_dump = validated_dump(OfflineNeuDestInfoV2, dest_info)
        half_attrs_dump = validated_dump(OfflineNeuFullAttrsV2Part1, half_attrs)

        pkg_half_neu = OfflineFrameGenV2._gen_pkg_half_neu(
            dest_info_dump, half_attrs_dump
        )
        addr_core_xy, addr_core_x, addr_core_y = coordzxy_to_sign_magnitude(
            (
                dest_info["addr_core_xy"],
                dest_info["addr_core_x"],
                dest_info["addr_core_y"],
            )
        )
        addr_copy_xy, addr_copy_x, addr_copy_y = coordzxy_to_sign_magnitude(
            (
                dest_info["addr_copy_xy"],
                dest_info["addr_copy_x"],
                dest_info["addr_copy_y"],
            )
        )

        assert pkg_half_neu.dtype == FRAME_DTYPE
        assert pkg_half_neu.shape == (2,)
        assert bit_field(
            pkg_half_neu[0],
            Off_Cfg3_V2.Full.Word1.WEIGHT_SKEW_LOW5_OFFSET,
            Off_Cfg3_V2.Full.Word1.WEIGHT_SKEW_LOW5_MASK,
        ) == (half_attrs["weight_skew"] & Off_Cfg3_V2.Full.Word1.WEIGHT_SKEW_LOW5_MASK)
        assert bit_field(
            pkg_half_neu[1],
            Off_Cfg3_V2.Full.Word2.WEIGHT_SKEW_HIGH11_OFFSET,
            Off_Cfg3_V2.Full.Word2.WEIGHT_SKEW_HIGH11_MASK,
        ) == (half_attrs["weight_skew"] >> 5)
        assert (
            bit_field(
                pkg_half_neu[0],
                Off_Cfg3_V2.Full.Word1.WEIGHT_ADDRESS_START_OFFSET,
                Off_Cfg3_V2.Full.Word1.WEIGHT_ADDRESS_START_MASK,
            )
            == half_attrs["weight_address_start"]
        )
        assert (
            bit_field(
                pkg_half_neu[0],
                Off_Cfg3_V2.Full.Word1.WEIGHT_ADDRESS_END_OFFSET,
                Off_Cfg3_V2.Full.Word1.WEIGHT_ADDRESS_END_MASK,
            )
            == half_attrs["weight_address_end"]
        )
        assert bit_field(
            pkg_half_neu[1],
            Off_Cfg3_V2.Full.Word2.ADDR_CORE_Y_OFFSET,
            Off_Cfg3_V2.Full.Word2.ADDR_CORE_Y_MASK,
        ) == addr_core_y
        assert bit_field(
            pkg_half_neu[1],
            Off_Cfg3_V2.Full.Word2.ADDR_CORE_XY_OFFSET,
            Off_Cfg3_V2.Full.Word2.ADDR_CORE_XY_MASK,
        ) == addr_core_xy
        assert bit_field(
            pkg_half_neu[1],
            Off_Cfg3_V2.Full.Word2.ADDR_CORE_X_OFFSET,
            Off_Cfg3_V2.Full.Word2.ADDR_CORE_X_MASK,
        ) == addr_core_x
        assert bit_field(
            pkg_half_neu[1],
            Off_Cfg3_V2.Full.Word2.ADDR_COPY_XY_OFFSET,
            Off_Cfg3_V2.Full.Word2.ADDR_COPY_XY_MASK,
        ) == addr_copy_xy
        assert bit_field(
            pkg_half_neu[1],
            Off_Cfg3_V2.Full.Word2.ADDR_COPY_X_OFFSET,
            Off_Cfg3_V2.Full.Word2.ADDR_COPY_X_MASK,
        ) == addr_copy_x
        assert bit_field(
            pkg_half_neu[1],
            Off_Cfg3_V2.Full.Word2.ADDR_COPY_Y_OFFSET,
            Off_Cfg3_V2.Full.Word2.ADDR_COPY_Y_MASK,
        ) == addr_copy_y

    def test_gen_pkg_full_neu_fields(self):
        dest_info = build_v2_dest_info_params(
            tick_relative=9,
            addr_axon=17,
            addr_core_xy=1,
            addr_core_x=2,
            addr_core_y=3,
            addr_copy_xy=4,
            addr_copy_x=5,
            addr_copy_y=6,
        )
        full_attrs1 = build_v2_half_attrs_params(
            weight_skew=0x123,
            weight_address_start=10,
            weight_address_end=20,
            output_type=OutputType.POTENTIAL,
            fold_type=FoldType.UNFOLDED,
            neuron_type=NeuronType.FULL,
            vjt=0x2468ACE,
        )
        full_attrs2 = build_v2_full_attrs_part2_params(
            reset_mode=ResetMode.MODE_LINEAR,
            reset_v=-123,
            threshold_neg_mode=ThresholdNegMode.FLOOR,
            threshold_pos_mode=ThresholdPosMode.CEILING,
            threshold_neg=-0x12345,
            threshold_pos=0x12345678,
            lateral_inhibition=LateralInhibitionMode.ENABLE,
            leak_multi_sequence=LeakMultiComparisonOrder.AFTER_COMPARE,
            leak_multi_input=LeakMultiInputMode.ENABLE,
            leak_multi_mode=LeakMultiMode.ENABLE,
            leak_add_mode=LeakAddMode.BACKWARD,
            leak_tau=17,
            leak_v=-0x1234,
            weight_compress=WeightCompressType.SPARSE,
            vjt_initial=0x345,
        )

        dest_info_dump = validated_dump(OfflineNeuDestInfoV2, dest_info)
        full_attrs1_dump = validated_dump(OfflineNeuFullAttrsV2Part1, full_attrs1)
        full_attrs2_dump = validated_dump(OfflineNeuFullAttrsV2Part2, full_attrs2)

        pkg_full_neu = OfflineFrameGenV2._gen_pkg_full_neu(
            dest_info_dump, full_attrs1_dump, full_attrs2_dump
        )

        threshold_pos = (
            bit_field(
                pkg_full_neu[3],
                Off_Cfg3_V2.Full.Word4.THRESHOLD_POS_HIGH12_OFFSET,
                Off_Cfg3_V2.Full.Word4.THRESHOLD_POS_HIGH12_MASK,
            )
            << 20
        ) | bit_field(
            pkg_full_neu[2],
            Off_Cfg3_V2.Full.Word3.THRESHOLD_POS_LOW20_OFFSET,
            Off_Cfg3_V2.Full.Word3.THRESHOLD_POS_LOW20_MASK,
        )

        assert pkg_full_neu.dtype == FRAME_DTYPE
        assert pkg_full_neu.shape == (4,)
        assert threshold_pos == full_attrs2["threshold_pos"]
        assert bit_field(
            pkg_full_neu[3],
            Off_Cfg3_V2.Full.Word4.THRESHOLD_NEG_OFFSET,
            Off_Cfg3_V2.Full.Word4.THRESHOLD_NEG_MASK,
        ) == (full_attrs2["threshold_neg"] & Off_Cfg3_V2.Full.Word4.THRESHOLD_NEG_MASK)
        assert bit_field(
            pkg_full_neu[3],
            Off_Cfg3_V2.Full.Word4.RESET_V_OFFSET,
            Off_Cfg3_V2.Full.Word4.RESET_V_MASK,
        ) == (full_attrs2["reset_v"] & Off_Cfg3_V2.Full.Word4.RESET_V_MASK)
        assert bit_field(
            pkg_full_neu[2],
            Off_Cfg3_V2.Full.Word3.LEAK_V_OFFSET,
            Off_Cfg3_V2.Full.Word3.LEAK_V_MASK,
        ) == (full_attrs2["leak_v"] & Off_Cfg3_V2.Full.Word3.LEAK_V_MASK)
        assert (
            bit_field(
                pkg_full_neu[2],
                Off_Cfg3_V2.Full.Word3.WEIGHT_COMPRESS_OFFSET,
                Off_Cfg3_V2.Full.Word3.WEIGHT_COMPRESS_MASK,
            )
            == WeightCompressType.SPARSE.value
        )

    def test_gen_pkg_folded_neu_fields(self):
        attrs1 = build_v2_folded_attrs_part1_params(
            fold_range_xy=2,
            fold_range_x=3,
            fold_range_y=4,
            fold_skew_xy=5,
            fold_skew_x=6,
            fold_skew_y=0x155,
            fold_axon_xy=7,
            fold_axon_x=8,
            fold_axon_y=9,
            fold_number=24,
        )
        attrs2_1 = build_v2_folded_attrs_part2_params(
            fold_vjt_3=100,
            fold_vjt_2=200,
            fold_vjt_1=300,
            fold_vjt_0=400,
        )
        attrs2_2 = build_v2_folded_attrs_part2_params(
            fold_vjt_3=10,
            fold_vjt_2=20,
            fold_vjt_1=30,
            fold_vjt_0=40,
        )

        attrs1_dump = validated_dump(OfflineNeuFoldedAttrsV2Part1, attrs1)
        attrs2_1_dump = validated_dump(OfflineNeuFoldedAttrsV2Part2, attrs2_1)
        attrs2_2_dump = validated_dump(OfflineNeuFoldedAttrsV2Part2, attrs2_2)

        pkg_folded_neu = OfflineFrameGenV2._gen_pkg_folded_neu(
            attrs1_dump, attrs2_1_dump, attrs2_2_dump
        )

        assert pkg_folded_neu.dtype == FRAME_DTYPE
        assert pkg_folded_neu.shape == (6,)
        assert bit_field(
            pkg_folded_neu[0],
            Off_Cfg3_V2.Fold.Word1.FOLD_SKEW_Y_LOW2_OFFSET,
            Off_Cfg3_V2.Fold.Word1.FOLD_SKEW_Y_LOW2_MASK,
        ) == (attrs1["fold_skew_y"] & Off_Cfg3_V2.Fold.Word1.FOLD_SKEW_Y_LOW2_MASK)
        assert bit_field(
            pkg_folded_neu[1],
            Off_Cfg3_V2.Fold.Word2.FOLD_SKEW_Y_HIGH9_OFFSET,
            Off_Cfg3_V2.Fold.Word2.FOLD_SKEW_Y_HIGH9_MASK,
        ) == (attrs1["fold_skew_y"] >> 2)
        assert (
            bit_field(
                pkg_folded_neu[2],
                Off_Cfg3_V2.Fold.Word3.FOLD_VJT_0_OFFSET,
                Off_Cfg3_V2.Fold.Word3.FOLD_VJT_0_MASK,
            )
            == attrs2_1["fold_vjt_0"]
        )
        assert (
            bit_field(
                pkg_folded_neu[3],
                Off_Cfg3_V2.Fold.Word4.FOLD_VJT_3_OFFSET,
                Off_Cfg3_V2.Fold.Word4.FOLD_VJT_3_MASK,
            )
            == attrs2_1["fold_vjt_3"]
        )
        assert (
            bit_field(
                pkg_folded_neu[4],
                Off_Cfg3_V2.Fold.Word3.FOLD_VJT_1_OFFSET,
                Off_Cfg3_V2.Fold.Word3.FOLD_VJT_1_MASK,
            )
            == attrs2_2["fold_vjt_1"]
        )
        assert (
            bit_field(
                pkg_folded_neu[5],
                Off_Cfg3_V2.Fold.Word4.FOLD_VJT_2_OFFSET,
                Off_Cfg3_V2.Fold.Word4.FOLD_VJT_2_MASK,
            )
            == attrs2_2["fold_vjt_2"]
        )

    @pytest.mark.parametrize(
        "wrapper_name,args,expected_size",
        [
            (
                "gen_config_frame3_pkg_half",
                (
                    build_v2_dest_info_params(),
                    build_v2_half_attrs_params(),
                ),
                2,
            ),
            (
                "gen_config_frame3_pkg_full",
                (
                    build_v2_dest_info_params(),
                    build_v2_half_attrs_params(),
                    build_v2_full_attrs_part2_params(),
                ),
                4,
            ),
            (
                "gen_config_frame3_pkg_folded",
                (
                    build_v2_folded_attrs_part1_params(),
                    [build_v2_folded_attrs_part2_params()],
                ),
                4,
            ),
        ],
        ids=["half", "full", "folded"],
    )
    def test_gen_config_frame3_pkg_wrappers(self, wrapper_name, args, expected_size):
        result = getattr(OfflineFrameGenV2, wrapper_name)(*args)

        assert result.size == expected_size

    @pytest.mark.parametrize(
        "kwargs,expected_sizes",
        [
            (
                {
                    "dest_info": build_v2_dest_info_params(),
                    "full_attrs1": build_v2_half_attrs_params(),
                    "full_attrs2": build_v2_full_attrs_part2_params(),
                    "folded_attrs1": build_v2_folded_attrs_part1_params(),
                    "folded_attrs2_": [
                        build_v2_folded_attrs_part2_params(),
                        build_v2_folded_attrs_part2_params(fold_vjt_0=10),
                    ],
                },
                [2, 4, 6],
            ),
            (
                {
                    "dest_info": build_v2_dest_info_params(),
                    "full_attrs1": None,
                    "full_attrs2": None,
                    "folded_attrs1": build_v2_folded_attrs_part1_params(),
                    "folded_attrs2_": [build_v2_folded_attrs_part2_params()],
                },
                [0, 0, 4],
            ),
        ],
        ids=["all", "folded-only"],
    )
    def test_gen_config_frame3_pkg_neu_valid_combinations(self, kwargs, expected_sizes):
        result = OfflineFrameGenV2.gen_config_frame3_pkg_neu(**kwargs)

        assert [pkg.size for pkg in result] == expected_sizes

    @pytest.mark.parametrize(
        "kwargs,error_match",
        [
            (
                {
                    "dest_info": build_v2_dest_info_params(),
                    "full_attrs1": None,
                    "full_attrs2": build_v2_full_attrs_part2_params(),
                    "folded_attrs1": None,
                    "folded_attrs2_": [],
                },
                "full neuron.*missing part1",
            ),
            (
                {
                    "dest_info": build_v2_dest_info_params(),
                    "full_attrs1": build_v2_half_attrs_params(),
                    "full_attrs2": None,
                    "folded_attrs1": build_v2_folded_attrs_part1_params(),
                    "folded_attrs2_": [],
                },
                "folded neuron.*missing part2",
            ),
            (
                {
                    "dest_info": build_v2_dest_info_params(),
                    "full_attrs1": build_v2_half_attrs_params(),
                    "full_attrs2": None,
                    "folded_attrs1": None,
                    "folded_attrs2_": [build_v2_folded_attrs_part2_params()],
                },
                "folded neuron.*missing part1",
            ),
        ],
        ids=["full-missing-part1", "folded-missing-part2", "folded-missing-part1"],
    )
    def test_gen_config_frame3_pkg_neu_rejects_invalid_combinations(
        self, kwargs, error_match
    ):
        with pytest.raises(ValueError, match=error_match):
            OfflineFrameGenV2.gen_config_frame3_pkg_neu(**kwargs)

    @pytest.mark.parametrize(
        "weight_width, input_width, csc_compress",
        [
            (8, 8, False),
            (DataWidth.WIDTH_4BIT, DataWidth.WIDTH_2BIT, False),
            (DataWidth.WIDTH_8BIT, DataWidth.WIDTH_1BIT, CSCAccelerateMode.ENABLE),
        ],
    )
    def test_gen_config_frame3_weight_pkg(
        self,
        weight_width: DataWidthLE8Like,
        input_width: DataWidthLE8Like,
        csc_compress: bool | CSCAccelerateMode,
        fixed_rng: np.random.Generator,
    ):
        is_sparse = (
            csc_compress != CSCAccelerateMode.DISABLE and csc_compress is not False
        )
        width_bits = normalize_width_bits(weight_width)
        weight = gen_v2_weight_array(
            32, width_bits, True, fixed_rng, sparse_ratio=0.4 if is_sparse else 0.0
        )

        result = OfflineFrameGenV2.gen_config_frame3_weight_pkg(
            weight, weight_width, input_width, csc_compress
        )

        if is_sparse:
            input_bits = normalize_width_bits(input_width)
            expected = weight_csc_pack(weight, width_bits, input_bits)
        else:
            expected = weight_dense_pack(weight, width_bits)

        assert np.array_equal(result, expected)

    @pytest.mark.parametrize(
        "weight_width, input_width",
        [
            (DataWidth.WIDTH_16BIT, DataWidth.WIDTH_1BIT),
            (DataWidth.WIDTH_1BIT, DataWidth.WIDTH_32BIT),
        ],
    )
    def test_gen_config_frame3_weight_pkg_rejects_unsupported_widths(
        self, weight_width, input_width
    ):
        with pytest.raises(ValueError, match="1/2/4/8-bit"):
            OfflineFrameGenV2.gen_config_frame3_weight_pkg(
                np.ones(8, dtype=np.int8), weight_width, input_width
            )

    @pytest.mark.parametrize(
        "ts, axon, data, target_lcn",
        [
            (
                np.array([0b0011_0100, 0b0011_0100]),
                np.array([0b0_0100_1011_0011, 0b100_1011_0011]),
                np.array([-16, 1], dtype=np.int8),
                LCN_EX.LCN_4X,
            ),
            (
                0b1011_0100,
                0b1_1011_0011,
                np.array([10], dtype=np.uint8),
                LCN_EX.LCN_1X,
            ),
        ],
    )
    def test_wf1(self, v2_packet_route, ts, axon, data, target_lcn):
        pkt_offset, pkt_ncopy = v2_packet_route
        wf1 = OfflineFrameGenV2.gen_work_frame1(
            pkt_offset, pkt_ncopy, ts, axon, target_lcn, data
        )

        ts0 = ts[0] if isinstance(ts, np.ndarray) else ts
        axon0 = axon[0] if isinstance(axon, np.ndarray) else axon
        ts_width, ax_width = OfflineFrameGenV2.LCN_TO_TS_AXON_WIDTHS[target_lcn.value]

        assert wf1.dtype == FRAME_DTYPE
        assert bit_field(
            wf1[0],
            Off_Work1_V2.TIMESTEP_HIGH7_OFFSET,
            Off_Work1_V2.TIMESTEP_HIGH7_MASK,
        ) == ((ts0 >> (ts_width - 1)) & Off_Work1_V2.TIMESTEP_HIGH7_MASK)
        assert (
            bit_field(wf1[0], Off_Work1_V2.AXON_ADDR_OFFSET, _mask(ax_width)) == axon0
        )

    def test_wf1_filters_zero_payloads(self, v2_packet_route):
        pkt_offset, pkt_ncopy = v2_packet_route
        wf1 = OfflineFrameGenV2.gen_work_frame1(
            pkt_offset,
            pkt_ncopy,
            np.array([1, 2, 3, 4]),
            np.array([10, 11, 12, 13]),
            LCN_EX.LCN_1X,
            np.array([0, 5, 0, 6], dtype=np.uint8),
        )

        assert wf1.shape == (2,)
        assert bit_field(wf1[0], Off_Work1_V2.DATA_OFFSET, Off_Work1_V2.DATA_MASK) == 5
        assert bit_field(wf1[1], Off_Work1_V2.DATA_OFFSET, Off_Work1_V2.DATA_MASK) == 6

    @pytest.mark.parametrize("data_dtype", [np.uint8, np.int8])
    def test_wf1_generic(self, v2_packet_route, data_dtype):
        pkt_offset, pkt_ncopy = v2_packet_route
        target_lcn = LCN_EX.LCN_4X
        n = 512

        ts_width, ax_width = OfflineFrameGenV2.LCN_TO_TS_AXON_WIDTHS[
            _target_lcn_index(target_lcn)
        ]
        timesteps = (np.arange(n, dtype=np.uint32) * 3 + 5) & _mask(ts_width)
        axons = (np.arange(n, dtype=np.uint32) * 7 + 11) & _mask(ax_width)

        data = gen_random_array((n,), data_dtype, sparse_ratio=1 / 6)
        if np.issubdtype(np.dtype(data_dtype), np.signedinteger):
            data[1] = np.iinfo(data_dtype).min
            data[2] = -1
        else:
            data[1] = np.iinfo(data_dtype).max
        data[0] = 0

        wf1 = OfflineFrameGenV2.gen_work_frame1(
            pkt_offset, pkt_ncopy, timesteps, axons, target_lcn, data
        )

        mask = np.flatnonzero(data)
        expected_data = data.view(np.uint8).ravel()[mask]
        expected_ts_ax_addr = _offline_work_ts_ax_addr(
            timesteps, axons, target_lcn, Off_Work1_V2
        )[mask]
        expected_frames = (
            get_frame_dest_v2(FH.WORK_TYPE1, pkt_offset, pkt_ncopy)
            + expected_ts_ax_addr
            + expected_data
        ).astype(FRAME_DTYPE)

        assert wf1.shape == (mask.size,)
        assert np.array_equal(wf1, expected_frames)
        assert np.any(data == 0)
        assert np.count_nonzero(data) == wf1.size
        assert np.all(expected_data != 0)
        if np.issubdtype(np.dtype(data_dtype), np.signedinteger):
            assert np.any(data < 0)
        assert np.array_equal(wf1 & Off_Work1_V2.DATA_MASK, expected_data)

    def test_wf2_packs_int32_voltage_lsb_first(self, v2_packet_route):
        pkt_offset, pkt_ncopy = v2_packet_route
        wf2 = OfflineFrameGenV2.gen_work_frame2(
            pkt_offset,
            pkt_ncopy,
            np.array([0x83, 0x04, 0x05]),
            np.array([0x106, 0x107, 0x108]),
            0,
            np.array([0x12345678, 0, -2], dtype=np.int32),
        )

        expected_bytes = np.array(
            [0x78, 0x56, 0x34, 0x12, 0xFE, 0xFF, 0xFF, 0xFF], dtype=np.uint8
        )
        expected_ts_ax_addr = _offline_work_ts_ax_addr(
            np.array([0x83, 0x04, 0x05]),
            np.array([0x106, 0x107, 0x108]),
            0,
            Off_Work2_V2,
        )
        expected_frames = (
            get_frame_dest_v2(FH.WORK_TYPE2, pkt_offset, pkt_ncopy)
            + np.repeat(expected_ts_ax_addr[[0, 2]], 4)
            + expected_bytes
        ).astype(FRAME_DTYPE)

        assert wf2.shape == (8,)
        assert np.array_equal(wf2, expected_frames)
        assert np.array_equal(wf2 & Off_Work2_V2.VJT_MASK, expected_bytes)

    def test_wf2_generic(self, v2_packet_route):
        pkt_offset, pkt_ncopy = v2_packet_route
        target_lcn = LCN_EX.LCN_8X.value
        n = 384

        ts_width, ax_width = OfflineFrameGenV2.LCN_TO_TS_AXON_WIDTHS[
            _target_lcn_index(target_lcn)
        ]
        timesteps = (np.arange(n, dtype=np.uint32) * 5 + 1) & _mask(ts_width)
        axons = (np.arange(n, dtype=np.uint32) * 9 + 3) & _mask(ax_width)

        voltage = gen_random_array((n,), np.int32, sparse_ratio=1 / 8)
        voltage[1] = np.iinfo(np.int32).min
        voltage[2] = np.iinfo(np.int32).max
        voltage[3] = -1
        voltage[0] = 0

        wf2 = OfflineFrameGenV2.gen_work_frame2(
            pkt_offset, pkt_ncopy, timesteps, axons, target_lcn, voltage
        )

        mask = np.flatnonzero(voltage)
        expected_bytes = (
            np.ascontiguousarray(
                np.asarray(voltage, dtype=np.dtype("<i4"))[mask], dtype=np.dtype("<i4")
            )
            .view(np.uint8)
            .reshape(-1)
        )
        expected_ts_ax_addr = np.repeat(
            _offline_work_ts_ax_addr(timesteps, axons, target_lcn, Off_Work2_V2)[mask],
            4,
        )
        expected_frames = (
            get_frame_dest_v2(FH.WORK_TYPE2, pkt_offset, pkt_ncopy)
            + expected_ts_ax_addr
            + expected_bytes
        ).astype(FRAME_DTYPE)

        assert wf2.shape == (mask.size * 4,)
        assert np.array_equal(wf2, expected_frames)
        assert np.count_nonzero(voltage) * 4 == wf2.size
        assert np.array_equal(wf2 & Off_Work2_V2.VJT_MASK, expected_bytes)

    def test_wf2_filters_zero_voltage(self, v2_packet_route):
        pkt_offset, pkt_ncopy = v2_packet_route
        wf2 = OfflineFrameGenV2.gen_work_frame2(
            pkt_offset,
            pkt_ncopy,
            np.array([1, 2]),
            np.array([10, 11]),
            LCN_EX.LCN_1X,
            np.array([0, 0], dtype=np.int32),
        )
        assert wf2.size == 0

    def test_control_frames(self, v2_packet_route):
        pkt_offset, pkt_ncopy = v2_packet_route
        ctrl1 = OfflineFrameGenV2.gen_control_frame1(pkt_offset, pkt_ncopy, 123)
        ctrl2 = OfflineFrameGenV2.gen_control_frame2(pkt_offset, pkt_ncopy)
        complete = OfflineFrameGenV2.gen_control_frame3(pkt_offset, pkt_ncopy, 7)

        assert single_frame_header_check(ctrl1[0], FH.CTRL_TYPE1)
        assert single_frame_header_check(ctrl2[0], FH.CTRL_TYPE2)
        assert single_frame_header_check(complete[0], FH.CTRL_TYPE3)
        assert (
            bit_field(ctrl1[0], FFV2.GENERAL_PAYLOAD_OFFSET, FFV2.GENERAL_PAYLOAD_MASK)
            == 123
        )
        assert (
            bit_field(
                complete[0], FFV2.GENERAL_PAYLOAD_OFFSET, FFV2.GENERAL_PAYLOAD_MASK
            )
            == 7
        )


class TestOnlineFrameGenV2:
    def test_cf1(self, v2_packet_route):
        pkt_offset, pkt_ncopy = v2_packet_route
        core_reg = OnlineCoreRegV2.model_validate(
            build_online_v2_core_reg_params(), strict=True
        )

        frames = OnlineFrameGenV2.gen_config_frame1(pkt_offset, core_reg, pkt_ncopy, 0)
        update_core_xy, update_core_x, update_core_y = coordzxy_to_sign_magnitude(
            (core_reg.update_core_xy, core_reg.update_core_x, core_reg.update_core_y)
        )
        test_core_xy, test_core_x, test_core_y_expected = coordzxy_to_sign_magnitude(
            (core_reg.test_core_xy, core_reg.test_core_x, core_reg.test_core_y)
        )

        neuron_number = (
            bit_field(
                frames[1],
                On_Cfg1_V2.Word1.NEURON_NUMBER_HIGH10_OFFSET,
                On_Cfg1_V2.Word1.NEURON_NUMBER_HIGH10_MASK,
            )
            << 3
        ) | bit_field(
            frames[2],
            On_Cfg1_V2.Word2.NEURON_NUMBER_LOW3_OFFSET,
            On_Cfg1_V2.Word2.NEURON_NUMBER_LOW3_MASK,
        )
        scale_out_bits = (
            bit_field(
                frames[2],
                On_Cfg1_V2.Word2.SCALE_OUT_HIGH15_OFFSET,
                On_Cfg1_V2.Word2.SCALE_OUT_HIGH15_MASK,
            )
            << 1
        ) | bit_field(
            frames[3],
            On_Cfg1_V2.Word3.SCALE_OUT_LOW1_OFFSET,
            On_Cfg1_V2.Word3.SCALE_OUT_LOW1_MASK,
        )
        test_core_y = (
            bit_field(
                frames[3],
                On_Cfg1_V2.Word3.TEST_CORE_Y_HIGH1_OFFSET,
                On_Cfg1_V2.Word3.TEST_CORE_Y_HIGH1_MASK,
            )
            << 5
        ) | bit_field(
            frames[4],
            On_Cfg1_V2.Word4.TEST_CORE_Y_LOW5_OFFSET,
            On_Cfg1_V2.Word4.TEST_CORE_Y_LOW5_MASK,
        )

        assert frames.dtype == FRAME_DTYPE
        assert frames.size == 6
        assert parse_package_header(frames) == (
            FramePackageType.CONF_TESTOUT.value,
            0,
            5,
        )
        assert (
            bit_field(
                frames[1],
                On_Cfg1_V2.Word1.WORK_MODE_OFFSET,
                On_Cfg1_V2.Word1.WORK_MODE_MASK,
            )
            == core_reg.work_mode
        )
        assert neuron_number == core_reg.neuron_number
        assert scale_out_bits == pack_bf16_scalar_bits(core_reg.scale_out)
        assert bit_field(
            frames[2],
            On_Cfg1_V2.Word2.SCALE_IN_OFFSET,
            On_Cfg1_V2.Word2.SCALE_IN_MASK,
        ) == pack_bf16_scalar_bits(core_reg.scale_in)
        assert bit_field(
            frames[3],
            On_Cfg1_V2.Word3.BIAS_OUT_OFFSET,
            On_Cfg1_V2.Word3.BIAS_OUT_MASK,
        ) == pack_bf16_scalar_bits(core_reg.bias_out)
        assert bit_field(
            frames[3],
            On_Cfg1_V2.Word3.LEARNING_RATE_OFFSET,
            On_Cfg1_V2.Word3.LEARNING_RATE_MASK,
        ) == pack_bf16_scalar_bits(core_reg.learning_rate)
        assert (
            bit_field(
                frames[3],
                On_Cfg1_V2.Word3.UPDATE_CORE_XY_OFFSET,
                On_Cfg1_V2.Word3.UPDATE_CORE_XY_MASK,
            )
            == update_core_xy
        )
        assert (
            bit_field(
                frames[3],
                On_Cfg1_V2.Word3.UPDATE_CORE_Y_OFFSET,
                On_Cfg1_V2.Word3.UPDATE_CORE_Y_MASK,
            )
            == update_core_y
        )
        assert (
            bit_field(
                frames[3],
                On_Cfg1_V2.Word3.TEST_CORE_XY_OFFSET,
                On_Cfg1_V2.Word3.TEST_CORE_XY_MASK,
            )
            == test_core_xy
        )
        assert (
            bit_field(
                frames[3],
                On_Cfg1_V2.Word3.TEST_CORE_X_OFFSET,
                On_Cfg1_V2.Word3.TEST_CORE_X_MASK,
            )
            == test_core_x
        )
        assert test_core_y == test_core_y_expected
        assert (
            bit_field(
                frames[5],
                On_Cfg1_V2.Word5.TICK_DURATION_OFFSET,
                On_Cfg1_V2.Word5.TICK_DURATION_MASK,
            )
            == core_reg.tick_duration
        )

    @pytest.mark.parametrize(
        "act_dtype", [np.int16, np.float16, np.float32, np.float64]
    )
    def test_cf2(self, v2_packet_route, act_dtype):
        pkt_offset, _ = v2_packet_route
        arr_pot = np.arange(-128, 128, dtype=np.int32)
        if np.issubdtype(np.dtype(act_dtype), np.floating):
            arr_act = (
                (np.arange(256, dtype=np.float32) / 16) - np.float32(8.0)
            ).astype(act_dtype)
        else:
            arr_act = np.arange(-128, 128, dtype=act_dtype)

        frames = OnlineFrameGenV2.gen_config_frame2(pkt_offset, arr_pot, arr_act)

        assert frames.size == 257

        arr_pot2, arr_act_bits = extract_online_lut_from_cf2(frames)
        assert np.array_equal(arr_pot2, arr_pot)
        if np.issubdtype(np.dtype(act_dtype), np.floating):
            expected_act_bits = pack_bf16_payload_bits(arr_act)
            assert np.array_equal(arr_act_bits, expected_act_bits)
        else:
            assert np.array_equal(arr_act_bits.view(act_dtype), arr_act)

    @pytest.mark.parametrize("n_package", [0, 1, 100, 1000, 8000])
    def test_gen_config_frame3_pkg_header(self, v2_packet_route, n_package):
        pkt_offset, _ = v2_packet_route
        frames = OnlineFrameGenV2.gen_config_frame3_pkg_header(
            pkt_offset, 12, n_package
        )

        assert frames.dtype == FRAME_DTYPE
        assert frames.shape == (1,)
        assert single_frame_header_check(frames[0], FH.CONFIG_TYPE3)
        assert parse_package_header(frames) == (
            FramePackageType.CONF_TESTOUT.value,
            12,
            n_package,
        )

    def test_gen_pkg_half_neu_fields(self):
        dest_info = build_v2_dest_info_params(
            tick_relative=0x7E,
            addr_axon=0xAB,
            addr_core_xy=-1,
            addr_core_x=2,
            addr_core_y=-3,
            addr_copy_xy=4,
            addr_copy_x=-5,
            addr_copy_y=6,
        )
        half_attrs = build_online_v2_half_attrs_params(
            weight_skew=0xABC,
            weight_address_start=0x123,
            weight_address_end=0x456,
            output_type=OnlineOutputType.POTENTIAL,
            fold_type=FoldType.UNFOLDED,
            neuron_type=NeuronType.FULL,
            vjt=np.float32(1.25),
        )

        dest_info_dump = validated_dump(OnlineNeuDestInfoV2, dest_info)
        half_attrs_dump = validated_dump(OnlineNeuHalfAttrsV2, half_attrs)

        pkg_half_neu = OnlineFrameGenV2._gen_pkg_half_neu(
            dest_info_dump, half_attrs_dump
        )
        _, _, addr_core_y = coordzxy_to_sign_magnitude(
            (
                dest_info["addr_core_xy"],
                dest_info["addr_core_x"],
                dest_info["addr_core_y"],
            )
        )
        _, addr_copy_x, _ = coordzxy_to_sign_magnitude(
            (
                dest_info["addr_copy_xy"],
                dest_info["addr_copy_x"],
                dest_info["addr_copy_y"],
            )
        )

        assert pkg_half_neu.dtype == FRAME_DTYPE
        assert pkg_half_neu.shape == (2,)
        assert bit_field(
            pkg_half_neu[0],
            On_Cfg3_V2.Full.Word1.WEIGHT_SKEW_LOW4_OFFSET,
            On_Cfg3_V2.Full.Word1.WEIGHT_SKEW_LOW4_MASK,
        ) == (half_attrs["weight_skew"] & On_Cfg3_V2.Full.Word1.WEIGHT_SKEW_LOW4_MASK)
        assert bit_field(
            pkg_half_neu[1],
            On_Cfg3_V2.Full.Word2.WEIGHT_SKEW_HIGH12_OFFSET,
            On_Cfg3_V2.Full.Word2.WEIGHT_SKEW_HIGH12_MASK,
        ) == (half_attrs["weight_skew"] >> 4)
        assert bit_field(
            pkg_half_neu[0],
            On_Cfg3_V2.Full.Word1.VJT_OFFSET,
            On_Cfg3_V2.Full.Word1.VJT_MASK,
        ) == pack_fp32_scalar_bits(half_attrs["vjt"])
        assert (
            bit_field(
                pkg_half_neu[1],
                On_Cfg3_V2.Full.Word2.ADDR_CORE_Y_OFFSET,
                On_Cfg3_V2.Full.Word2.ADDR_CORE_Y_MASK,
            )
            == addr_core_y
        )
        assert (
            bit_field(
                pkg_half_neu[1],
                On_Cfg3_V2.Full.Word2.ADDR_COPY_X_OFFSET,
                On_Cfg3_V2.Full.Word2.ADDR_COPY_X_MASK,
            )
            == addr_copy_x
        )

    def test_gen_pkg_full_neu_fields(self):
        dest_info = build_v2_dest_info_params(
            tick_relative=9,
            addr_axon=17,
            addr_core_xy=1,
            addr_core_x=2,
            addr_core_y=3,
            addr_copy_xy=4,
            addr_copy_x=5,
            addr_copy_y=6,
        )
        full_attrs1 = build_online_v2_half_attrs_params(
            weight_skew=0x123,
            weight_address_start=10,
            weight_address_end=20,
            output_type=OnlineOutputType.POTENTIAL,
            fold_type=FoldType.UNFOLDED,
            neuron_type=NeuronType.FULL,
            vjt=np.float32(2.5),
        )
        full_attrs2 = build_v2_full_attrs_part2_params(
            reset_mode=ResetMode.MODE_LINEAR,
            reset_v=np.float16(-0.75),
            threshold_neg_mode=ThresholdNegMode.FLOOR,
            threshold_pos_mode=ThresholdPosMode.CEILING,
            threshold_neg=np.float32(-2.5),
            threshold_pos=np.float32(3.25),
            lateral_inhibition=LateralInhibitionMode.ENABLE,
            leak_multi_sequence=LeakMultiComparisonOrder.AFTER_COMPARE,
            leak_multi_input=LeakMultiInputMode.ENABLE,
            leak_multi_mode=LeakMultiMode.ENABLE,
            leak_add_mode=LeakAddMode.BACKWARD,
            leak_tau=17,
            leak_v=np.float16(-1.5),
            weight_compress=WeightCompressType.SPARSE,
            vjt_initial=np.float16(0.5),
        )

        dest_info_dump = validated_dump(OnlineNeuDestInfoV2, dest_info)
        full_attrs1_dump = validated_dump(OnlineNeuHalfAttrsV2, full_attrs1)
        full_attrs2_dump = validated_dump(OnlineNeuFullAttrsV2Part2, full_attrs2)

        pkg_full_neu = OnlineFrameGenV2._gen_pkg_full_neu(
            dest_info_dump, full_attrs1_dump, full_attrs2_dump
        )

        threshold_pos_bits = (
            bit_field(
                pkg_full_neu[3],
                On_Cfg3_V2.Full.Word4.THRESHOLD_POS_HIGH12_OFFSET,
                On_Cfg3_V2.Full.Word4.THRESHOLD_POS_HIGH12_MASK,
            )
            << 20
        ) | bit_field(
            pkg_full_neu[2],
            On_Cfg3_V2.Full.Word3.THRESHOLD_POS_LOW20_OFFSET,
            On_Cfg3_V2.Full.Word3.THRESHOLD_POS_LOW20_MASK,
        )

        assert pkg_full_neu.dtype == FRAME_DTYPE
        assert pkg_full_neu.shape == (4,)
        assert threshold_pos_bits == pack_fp32_scalar_bits(full_attrs2["threshold_pos"])
        assert bit_field(
            pkg_full_neu[3],
            On_Cfg3_V2.Full.Word4.THRESHOLD_NEG_OFFSET,
            On_Cfg3_V2.Full.Word4.THRESHOLD_NEG_MASK,
        ) == pack_fp32_scalar_bits(full_attrs2["threshold_neg"])
        assert bit_field(
            pkg_full_neu[3],
            On_Cfg3_V2.Full.Word4.RESET_V_OFFSET,
            On_Cfg3_V2.Full.Word4.RESET_V_MASK,
        ) == pack_bf16_scalar_bits(full_attrs2["reset_v"])
        assert bit_field(
            pkg_full_neu[2],
            On_Cfg3_V2.Full.Word3.LEAK_V_OFFSET,
            On_Cfg3_V2.Full.Word3.LEAK_V_MASK,
        ) == pack_bf16_scalar_bits(full_attrs2["leak_v"])
        assert bit_field(
            pkg_full_neu[2],
            On_Cfg3_V2.Full.Word3.VJT_INITIAL_OFFSET,
            On_Cfg3_V2.Full.Word3.VJT_INITIAL_MASK,
        ) == pack_bf16_scalar_bits(full_attrs2["vjt_initial"])
        assert (
            bit_field(
                pkg_full_neu[2],
                On_Cfg3_V2.Full.Word3.WEIGHT_COMPRESS_OFFSET,
                On_Cfg3_V2.Full.Word3.WEIGHT_COMPRESS_MASK,
            )
            == WeightCompressType.SPARSE.value
        )

    @pytest.mark.parametrize("float_dtype", [np.float16, np.float32, np.float64])
    def test_gen_pkg_full_neu_bf16_fields_numpy_float_inputs(self, float_dtype):
        dest_info_dump = validated_dump(
            OnlineNeuDestInfoV2, build_v2_dest_info_params()
        )
        full_attrs1_dump = validated_dump(
            OnlineNeuHalfAttrsV2,
            build_online_v2_half_attrs_params(
                neuron_type=NeuronType.FULL,
                vjt=np.float32(0.5),
            ),
        )
        full_attrs2 = build_v2_full_attrs_part2_params(
            reset_v=float_dtype(-0.75),
            threshold_neg=np.float32(-2.5),
            threshold_pos=np.float32(3.25),
            leak_v=float_dtype(-1.5),
            vjt_initial=float_dtype(0.5),
        )
        full_attrs2_dump = validated_dump(OnlineNeuFullAttrsV2Part2, full_attrs2)

        pkg_full_neu = OnlineFrameGenV2._gen_pkg_full_neu(
            dest_info_dump, full_attrs1_dump, full_attrs2_dump
        )

        assert bit_field(
            pkg_full_neu[3],
            On_Cfg3_V2.Full.Word4.RESET_V_OFFSET,
            On_Cfg3_V2.Full.Word4.RESET_V_MASK,
        ) == pack_bf16_scalar_bits(full_attrs2["reset_v"])
        assert bit_field(
            pkg_full_neu[2],
            On_Cfg3_V2.Full.Word3.LEAK_V_OFFSET,
            On_Cfg3_V2.Full.Word3.LEAK_V_MASK,
        ) == pack_bf16_scalar_bits(full_attrs2["leak_v"])
        assert bit_field(
            pkg_full_neu[2],
            On_Cfg3_V2.Full.Word3.VJT_INITIAL_OFFSET,
            On_Cfg3_V2.Full.Word3.VJT_INITIAL_MASK,
        ) == pack_bf16_scalar_bits(full_attrs2["vjt_initial"])

    def test_gen_pkg_folded_neu_fields(self):
        attrs1 = build_v2_folded_attrs_part1_params(
            fold_range_xy=2,
            fold_range_x=3,
            fold_range_y=4,
            fold_skew_xy=5,
            fold_skew_x=6,
            fold_skew_y=0x155,
            fold_axon_xy=7,
            fold_axon_x=8,
            fold_axon_y=9,
            fold_number=24,
        )
        attrs2_1 = build_v2_folded_attrs_part2_params(
            fold_vjt_3=np.float32(1.0),
            fold_vjt_2=np.float32(-2.5),
            fold_vjt_1=np.float32(3.25),
            fold_vjt_0=np.float32(-4.75),
        )
        attrs2_2 = build_v2_folded_attrs_part2_params(
            fold_vjt_3=np.float32(10.0),
            fold_vjt_2=np.float32(20.0),
            fold_vjt_1=np.float32(30.0),
            fold_vjt_0=np.float32(40.0),
        )

        attrs1_dump = validated_dump(OnlineNeuFoldedAttrsV2Part1, attrs1)
        attrs2_1_dump = validated_dump(OnlineNeuFoldedAttrsV2Part2, attrs2_1)
        attrs2_2_dump = validated_dump(OnlineNeuFoldedAttrsV2Part2, attrs2_2)

        pkg_folded_neu = OnlineFrameGenV2._gen_pkg_folded_neu(
            attrs1_dump, attrs2_1_dump, attrs2_2_dump
        )

        assert pkg_folded_neu.dtype == FRAME_DTYPE
        assert pkg_folded_neu.shape == (6,)
        assert bit_field(
            pkg_folded_neu[0],
            On_Cfg3_V2.Fold.Word1.FOLD_SKEW_Y_LOW2_OFFSET,
            On_Cfg3_V2.Fold.Word1.FOLD_SKEW_Y_LOW2_MASK,
        ) == (attrs1["fold_skew_y"] & On_Cfg3_V2.Fold.Word1.FOLD_SKEW_Y_LOW2_MASK)
        assert bit_field(
            pkg_folded_neu[1],
            On_Cfg3_V2.Fold.Word2.FOLD_SKEW_Y_HIGH9_OFFSET,
            On_Cfg3_V2.Fold.Word2.FOLD_SKEW_Y_HIGH9_MASK,
        ) == (attrs1["fold_skew_y"] >> 2)
        assert bit_field(
            pkg_folded_neu[2],
            On_Cfg3_V2.Fold.Word3.FOLD_VJT_0_OFFSET,
            On_Cfg3_V2.Fold.Word3.FOLD_VJT_0_MASK,
        ) == pack_fp32_scalar_bits(attrs2_1["fold_vjt_0"])
        assert bit_field(
            pkg_folded_neu[3],
            On_Cfg3_V2.Fold.Word4.FOLD_VJT_3_OFFSET,
            On_Cfg3_V2.Fold.Word4.FOLD_VJT_3_MASK,
        ) == pack_fp32_scalar_bits(attrs2_1["fold_vjt_3"])
        assert bit_field(
            pkg_folded_neu[4],
            On_Cfg3_V2.Fold.Word3.FOLD_VJT_1_OFFSET,
            On_Cfg3_V2.Fold.Word3.FOLD_VJT_1_MASK,
        ) == pack_fp32_scalar_bits(attrs2_2["fold_vjt_1"])
        assert bit_field(
            pkg_folded_neu[5],
            On_Cfg3_V2.Fold.Word4.FOLD_VJT_2_OFFSET,
            On_Cfg3_V2.Fold.Word4.FOLD_VJT_2_MASK,
        ) == pack_fp32_scalar_bits(attrs2_2["fold_vjt_2"])

    @pytest.mark.parametrize(
        "wrapper_name,args,expected_size",
        [
            (
                "gen_config_frame3_pkg_half",
                (
                    build_v2_dest_info_params(),
                    build_online_v2_half_attrs_params(vjt=np.float32(0.5)),
                ),
                2,
            ),
            (
                "gen_config_frame3_pkg_full",
                (
                    build_v2_dest_info_params(),
                    build_online_v2_half_attrs_params(vjt=np.float32(0.5)),
                    build_v2_full_attrs_part2_params(
                        reset_v=np.float16(0.25),
                        threshold_neg=np.float32(-1.0),
                        threshold_pos=np.float32(1.0),
                        leak_v=np.float16(2.0),
                        vjt_initial=np.float16(0.75),
                    ),
                ),
                4,
            ),
            (
                "gen_config_frame3_pkg_folded",
                (
                    build_v2_folded_attrs_part1_params(),
                    [build_v2_folded_attrs_part2_params(fold_vjt_0=np.float32(1.0))],
                ),
                4,
            ),
        ],
        ids=["half", "full", "folded"],
    )
    def test_gen_config_frame3_pkg_wrappers(self, wrapper_name, args, expected_size):
        result = getattr(OnlineFrameGenV2, wrapper_name)(*args)

        assert result.size == expected_size

    def test_gen_config_frame3_pkg_neu_valid_combinations(self):
        result = OnlineFrameGenV2.gen_config_frame3_pkg_neu(
            build_v2_dest_info_params(),
            build_online_v2_half_attrs_params(vjt=np.float32(0.5)),
            build_v2_full_attrs_part2_params(
                reset_v=np.float16(0.25),
                threshold_neg=np.float32(-1.0),
                threshold_pos=np.float32(1.0),
                leak_v=np.float16(2.0),
                vjt_initial=np.float16(0.75),
            ),
            build_v2_folded_attrs_part1_params(),
            [
                build_v2_folded_attrs_part2_params(fold_vjt_0=np.float32(1.0)),
                build_v2_folded_attrs_part2_params(fold_vjt_0=np.float32(2.0)),
            ],
        )

        assert [pkg.size for pkg in result] == [2, 4, 6]

    @pytest.mark.parametrize(
        "args,error_match",
        [
            (
                (
                    build_v2_dest_info_params(),
                    None,
                    build_v2_full_attrs_part2_params(),
                    None,
                    [],
                ),
                "full neuron.*missing part1",
            ),
            (
                (
                    build_v2_dest_info_params(),
                    build_online_v2_half_attrs_params(),
                    None,
                    build_v2_folded_attrs_part1_params(),
                    [],
                ),
                "folded neuron.*missing part2",
            ),
            (
                (
                    build_v2_dest_info_params(),
                    build_online_v2_half_attrs_params(),
                    None,
                    None,
                    [build_v2_folded_attrs_part2_params()],
                ),
                "folded neuron.*missing part1",
            ),
        ],
        ids=["full-missing-part1", "folded-missing-part2", "folded-missing-part1"],
    )
    def test_gen_config_frame3_pkg_neu_rejects_invalid_combinations(
        self, args, error_match
    ):
        with pytest.raises(ValueError, match=error_match):
            OnlineFrameGenV2.gen_config_frame3_pkg_neu(*args)

    @pytest.mark.parametrize(
        "weight,csc_compress,expected_builder",
        [
            (
                np.array([1, -2, 3, 4, 5], dtype=np.int16),
                False,
                online_weight_dense_pack,
            ),
        ],
        ids=["dense"],
    )
    def test_gen_config_frame3_weight_pkg(self, weight, csc_compress, expected_builder):
        result = OnlineFrameGenV2.gen_config_frame3_weight_pkg(weight, csc_compress)
        expected = expected_builder(weight)

        assert np.array_equal(result, expected)

    @pytest.mark.parametrize("float_dtype", [np.float16, np.float32, np.float64])
    def test_gen_config_frame3_weight_pkg_csc_bf16(self, float_dtype):
        weight = np.array([0.0, 1.0, 0.0, -2.0, 3.25], dtype=float_dtype)
        result = OnlineFrameGenV2.gen_config_frame3_weight_pkg(
            weight, CSCAccelerateMode.ENABLE
        )

        packed_weights = np.ascontiguousarray(
            pack_bf16_payload_bits([1.0, -2.0, 3.25, 0.0]), dtype=np.dtype("<u2")
        ).view(FRAME_DTYPE)
        packed_indices = np.ascontiguousarray(
            np.array([1, 3, 4, 0], dtype=np.dtype("<u2"))
        ).view(FRAME_DTYPE)
        expected = np.array([packed_weights[0], packed_indices[0]], dtype=FRAME_DTYPE)

        assert result.shape == (2,)
        assert np.array_equal(result, expected)

    def test_online_weight_csc_pack_rejects_all_nonzero_unaligned(self):
        weight = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float16)
        with pytest.raises(ValueError, match="groups of 4"):
            OnlineFrameGenV2.gen_config_frame3_weight_pkg(
                weight, CSCAccelerateMode.ENABLE
            )

    @pytest.mark.parametrize("float_dtype", [np.float16, np.float32, np.float64])
    def test_online_weight_dense_pack_8_bf16_weights(self, float_dtype):
        weight = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=float_dtype)
        result = online_weight_dense_pack(weight)
        expected_bits = pack_bf16_payload_bits(weight).astype(FRAME_DTYPE, copy=False)
        shifts = 16 * np.arange(4, dtype=FRAME_DTYPE)
        expected = np.array(
            [
                np.bitwise_or.reduce(expected_bits[:4] << shifts, dtype=FRAME_DTYPE),
                np.bitwise_or.reduce(expected_bits[4:] << shifts, dtype=FRAME_DTYPE),
            ],
            dtype=FRAME_DTYPE,
        )

        assert result.shape == (2,)
        assert np.array_equal(result, expected)

    @pytest.mark.parametrize("float_dtype", [np.float16, np.float32, np.float64])
    def test_online_weight_dense_unpack(self, float_dtype):
        rng = np.random.default_rng(20260428)
        weight = rng.uniform(-8.0, 8.0, 77).astype(float_dtype)
        mapped = online_weight_dense_pack(weight)

        unpacked = online_weight_dense_unpack(mapped, weight.size)
        assert unpacked.dtype == np.uint16
        assert np.array_equal(unpacked, pack_bf16_payload_bits(weight).ravel())

    @pytest.mark.parametrize("float_dtype", [np.float16, np.float32, np.float64])
    def test_online_weight_csc_unpack(self, float_dtype):
        rng = np.random.default_rng(20260428)
        weight = rng.uniform(-8.0, 8.0, 77).astype(float_dtype)
        weight[rng.choice(weight.size, size=weight.size // 4, replace=False)] = 0
        mapped = online_weight_csc_pack(weight)

        unpacked = online_weight_csc_unpack(mapped, weight.size)
        assert unpacked.dtype == np.uint16
        assert np.array_equal(unpacked, pack_bf16_payload_bits(weight).ravel())

    def test_cf4(self, v2_packet_route):
        pkt_offset, pkt_ncopy = v2_packet_route
        input_array = np.arange(12, dtype=FRAME_DTYPE)

        frames = OnlineFrameGenV2.gen_config_frame4(
            pkt_offset, input_array, pkt_ncopy, 3
        )

        assert frames.dtype == FRAME_DTYPE
        assert frames.size == 13
        assert parse_package_header(frames) == (
            FramePackageType.CONF_TESTOUT.value,
            3,
            input_array.size,
        )
        assert np.array_equal(frames[1:], input_array)

    @pytest.mark.parametrize(
        "method_name,header,frame_format,target_lcn,data",
        _online_work_batch_cases(),
        ids=[
            "wf1-bool",
            "wf1-u1",
            "wf1-fp16",
            "wf2-fp16",
            "wf3-fp32",
            "wf3-fp16",
            "wf4-fp16",
        ],
    )
    def test_work_frames_generic(
        self, v2_packet_route, method_name, header, frame_format, target_lcn, data
    ):
        pkt_offset, pkt_ncopy = v2_packet_route
        n = data.size
        ts_width, ax_width = OnlineFrameGenV2.LCN_TO_TS_AXON_WIDTHS[
            _target_lcn_index(target_lcn)
        ]
        timesteps = (np.arange(n, dtype=np.uint32) * 5 + 3) & _mask(ts_width)
        axons = (np.arange(n, dtype=np.uint32) * 11 + 7) & _mask(ax_width)

        wf = getattr(OnlineFrameGenV2, method_name)(
            pkt_offset, pkt_ncopy, timesteps, axons, target_lcn, data
        )
        expected_frames, expected_bytes, expected_ts_ax_addr = (
            _expected_online_work_frames(
                header,
                frame_format,
                pkt_offset,
                pkt_ncopy,
                timesteps,
                axons,
                target_lcn,
                data,
            )
        )

        assert wf.shape == expected_frames.shape
        assert np.array_equal(wf, expected_frames)
        assert all(single_frame_header_check(frame, header) for frame in wf)
        payload_mask = (
            frame_format.DATA_MASK
            if hasattr(frame_format, "DATA_MASK")
            else frame_format.VJT_MASK
        )
        assert np.array_equal((wf & payload_mask).astype(np.uint8), expected_bytes)
        assert np.array_equal(
            (wf >> frame_format.TIMESTEP_AXON_OFFSET) & frame_format.TIMESTEP_AXON_MASK,
            expected_ts_ax_addr >> frame_format.TIMESTEP_AXON_OFFSET,
        )
        assert wf.size == np.count_nonzero(data) * data.dtype.itemsize
        assert np.any(data == 0)

    @pytest.mark.parametrize(
        "method_name,header,data,expected_bytes",
        [
            (
                "gen_work_frame1",
                FH.WORK_TYPE1,
                np.array([True, False]),
                np.array([0x01], dtype=np.uint8),
            ),
            (
                "gen_work_frame1",
                FH.WORK_TYPE1,
                np.array([0x01, 0x00], dtype=np.uint8),
                np.array([0x01], dtype=np.uint8),
            ),
            (
                "gen_work_frame2",
                FH.WORK_TYPE2,
                np.array([1.5, 0.0], dtype=np.float16),
                np.array([0x00, 0x3E], dtype=np.uint8),
            ),
            (
                "gen_work_frame3",
                FH.WORK_TYPE3,
                np.array([-2.5, 0.0], dtype=np.float32),
                np.array([0x00, 0x00, 0x20, 0xC0], dtype=np.uint8),
            ),
            (
                "gen_work_frame3",
                FH.WORK_TYPE3,
                np.array([-2.5, 0.0], dtype=np.float16),
                np.array([0x00, 0xC1], dtype=np.uint8),
            ),
            (
                "gen_work_frame4",
                FH.WORK_TYPE4,
                np.array([1.5, 0.0], dtype=np.float16),
                np.array([0x00, 0x3E], dtype=np.uint8),
            ),
        ],
        ids=["wf1-bool", "wf1-u1", "wf2-f16", "wf3-f32", "wf3-f16", "wf4-f16"],
    )
    def test_work_frames(
        self, v2_packet_route, method_name, header, data, expected_bytes
    ):
        pkt_offset, pkt_ncopy = v2_packet_route
        wf = getattr(OnlineFrameGenV2, method_name)(
            pkt_offset,
            pkt_ncopy,
            np.array([3, 4]),
            np.array([5, 6]),
            LCN_EX.LCN_1X,
            data,
        )

        assert wf.size == expected_bytes.size
        assert all(single_frame_header_check(frame, header) for frame in wf)
        assert np.array_equal(
            (wf & On_Work1_V2.DATA_MASK).astype(np.uint8), expected_bytes
        )
        assert np.all(
            ((wf >> On_Work1_V2.TIMESTEP_AXON_OFFSET) & On_Work1_V2.TIMESTEP_AXON_MASK)
            == ((3 << 8) | 5)
        )

    @pytest.mark.parametrize(
        "method_name,header,data,expected_bytes",
        [
            (
                "gen_work_frame1",
                FH.WORK_TYPE1,
                np.array([1.5, 0.0], dtype=np.float32),
                np.array([0x00, 0x3E], dtype=np.uint8),
            ),
            (
                "gen_work_frame2",
                FH.WORK_TYPE2,
                np.array([1.5, 0.0], dtype=np.float32),
                np.array([0x00, 0x3E], dtype=np.uint8),
            ),
            (
                "gen_work_frame3",
                FH.WORK_TYPE3,
                np.array([-2.5, 0.0], dtype=np.float64),
                np.array([0x00, 0x00, 0x20, 0xC0], dtype=np.uint8),
            ),
            (
                "gen_work_frame4",
                FH.WORK_TYPE4,
                np.array([1.5, 0.0], dtype=np.float32),
                np.array([0x00, 0x3E], dtype=np.uint8),
            ),
        ],
        ids=["wf1-f32-to-f16", "wf2-f32-to-f16", "wf3-f64-to-f32", "wf4-f32-to-f16"],
    )
    def test_work_frames_warn_on_float_truncation(
        self, v2_packet_route, method_name, header, data, expected_bytes
    ):
        pkt_offset, pkt_ncopy = v2_packet_route
        with pytest.warns(TruncationWarning, match="payload"):
            wf = getattr(OnlineFrameGenV2, method_name)(
                pkt_offset,
                pkt_ncopy,
                np.array([3, 4]),
                np.array([5, 6]),
                LCN_EX.LCN_1X,
                data,
            )

        assert wf.size == expected_bytes.size
        assert all(single_frame_header_check(frame, header) for frame in wf)
        assert np.array_equal(
            (wf & On_Work1_V2.DATA_MASK).astype(np.uint8), expected_bytes
        )
        assert np.all(
            ((wf >> On_Work1_V2.TIMESTEP_AXON_OFFSET) & On_Work1_V2.TIMESTEP_AXON_MASK)
            == ((3 << 8) | 5)
        )

    @pytest.mark.parametrize(
        "method_name,data,err_type,err_match",
        [
            ("gen_work_frame1", np.array([2], dtype=np.uint8), ValueError, "1-bit"),
            ("gen_work_frame1", np.array([1], dtype=np.int64), TypeError, "WF1"),
            ("gen_work_frame2", np.array([1], dtype=np.int16), TypeError, "WF2"),
            ("gen_work_frame3", np.array([1], dtype=np.int32), TypeError, "WF3"),
            ("gen_work_frame4", np.array([True]), TypeError, "WF4"),
        ],
        ids=["wf1-u8-not-u1", "wf1-int64", "wf2-int16", "wf3-int32", "wf4-bool"],
    )
    def test_work_frames_reject_invalid_payload_types(
        self, v2_packet_route, method_name, data, err_type, err_match
    ):
        pkt_offset, pkt_ncopy = v2_packet_route
        with pytest.raises(err_type, match=err_match):
            getattr(OnlineFrameGenV2, method_name)(
                pkt_offset,
                pkt_ncopy,
                np.array([3]),
                np.array([5]),
                LCN_EX.LCN_1X,
                data,
            )

    @pytest.mark.parametrize(
        "timesteps,axons,data,err_match",
        [
            (np.array([3, 4]), np.array([5]), np.ones(2, dtype=np.float16), "axons"),
            (
                np.array([3, 4]),
                np.array([5, 6]),
                np.ones(1, dtype=np.float16),
                "payload",
            ),
        ],
        ids=["axon-size", "payload-size"],
    )
    def test_work_frame_rejects_mismatched_lengths(
        self, v2_packet_route, timesteps, axons, data, err_match
    ):
        pkt_offset, pkt_ncopy = v2_packet_route
        with pytest.raises(ValueError, match=err_match):
            OnlineFrameGenV2.gen_work_frame2(
                pkt_offset, pkt_ncopy, timesteps, axons, LCN_EX.LCN_1X, data
            )

    def test_work_frame_payload_scalar_is_single_item(self, v2_packet_route):
        pkt_offset, pkt_ncopy = v2_packet_route
        wf = OnlineFrameGenV2.gen_work_frame2(
            pkt_offset,
            pkt_ncopy,
            np.array([3]),
            np.array([5]),
            LCN_EX.LCN_1X,
            np.float16(1.5),
        )

        assert wf.size == 2
        assert all(single_frame_header_check(frame, FH.WORK_TYPE2) for frame in wf)
        assert np.array_equal(
            (wf & On_Work1_V2.DATA_MASK).astype(np.uint8),
            np.array([0x00, 0x3E], dtype=np.uint8),
        )
        assert np.all(
            ((wf >> On_Work1_V2.TIMESTEP_AXON_OFFSET) & On_Work1_V2.TIMESTEP_AXON_MASK)
            == ((3 << 8) | 5)
        )

    def test_work_frame_payload_nd_is_flattened(self, v2_packet_route):
        pkt_offset, pkt_ncopy = v2_packet_route
        wf = OnlineFrameGenV2.gen_work_frame2(
            pkt_offset,
            pkt_ncopy,
            np.array([3, 4]),
            np.array([5, 6]),
            LCN_EX.LCN_1X,
            np.array([[1.5, 2.5]], dtype=np.float16),
        )

        assert wf.size == 4
        assert all(single_frame_header_check(frame, FH.WORK_TYPE2) for frame in wf)
        assert np.array_equal(
            (wf & On_Work1_V2.DATA_MASK).astype(np.uint8),
            np.array([0x00, 0x3E, 0x00, 0x41], dtype=np.uint8),
        )
        assert np.array_equal(
            ((wf >> On_Work1_V2.TIMESTEP_AXON_OFFSET) & On_Work1_V2.TIMESTEP_AXON_MASK),
            np.array([0x0305, 0x0305, 0x0406, 0x0406], dtype=FRAME_DTYPE),
        )

    @pytest.mark.parametrize(
        "method_name,data",
        [
            ("gen_work_frame2", np.array([-0.0], dtype=np.float16)),
            ("gen_work_frame3", np.array([-0.0], dtype=np.float32)),
            ("gen_work_frame4", np.array([-0.0], dtype=np.float16)),
        ],
        ids=["wf2-neg-zero", "wf3-neg-zero", "wf4-neg-zero"],
    )
    def test_work_frames_skip_negative_zero(self, v2_packet_route, method_name, data):
        pkt_offset, pkt_ncopy = v2_packet_route
        wf = getattr(OnlineFrameGenV2, method_name)(
            pkt_offset,
            pkt_ncopy,
            np.array([3]),
            np.array([5]),
            LCN_EX.LCN_1X,
            data,
        )
        assert wf.size == 0

    def test_control_frames(self, v2_packet_route):
        pkt_offset, pkt_ncopy = v2_packet_route
        ctrl1 = OnlineFrameGenV2.gen_control_frame1(pkt_offset, pkt_ncopy, 123)
        ctrl2 = OnlineFrameGenV2.gen_control_frame2(pkt_offset, pkt_ncopy)
        ctrl3 = OnlineFrameGenV2.gen_control_frame3(pkt_offset, pkt_ncopy, 7)

        ext_pkt_ncopy = AERPacketZXYCopy(4, 5, -2)
        ctrl4 = OnlineFrameGenV2.gen_control_frame4(
            pkt_offset, pkt_ncopy, ext_pkt_ncopy
        )
        ext_xy, ext_x, ext_y = ext_pkt_ncopy.to_sign_magnitude()
        expected_ext_payload = (
            (ext_xy << On_Ctrl4_V2.EXT_COPY_XY_ADDR_OFFSET)
            | (ext_x << On_Ctrl4_V2.EXT_COPY_X_ADDR_OFFSET)
            | (ext_y << On_Ctrl4_V2.EXT_COPY_Y_ADDR_OFFSET)
        )

        assert single_frame_header_check(ctrl1[0], FH.CTRL_TYPE1)
        assert single_frame_header_check(ctrl2[0], FH.CTRL_TYPE2)
        assert single_frame_header_check(ctrl3[0], FH.CTRL_TYPE3)
        assert single_frame_header_check(ctrl4[0], FH.CTRL_TYPE4)
        assert (
            bit_field(ctrl1[0], FFV2.GENERAL_PAYLOAD_OFFSET, FFV2.GENERAL_PAYLOAD_MASK)
            == 123
        )
        assert (
            bit_field(ctrl3[0], FFV2.GENERAL_PAYLOAD_OFFSET, FFV2.GENERAL_PAYLOAD_MASK)
            == 7
        )
        assert (
            bit_field(ctrl4[0], FFV2.GENERAL_PAYLOAD_OFFSET, FFV2.GENERAL_PAYLOAD_MASK)
            == expected_ext_payload
        )
        assert (
            bit_field(
                ctrl4[0],
                On_Ctrl4_V2.EXT_COPY_XY_ADDR_OFFSET,
                On_Ctrl4_V2.EXT_COPY_XY_ADDR_MASK,
            )
            == ext_xy
        )
        assert (
            bit_field(
                ctrl4[0],
                On_Ctrl4_V2.EXT_COPY_X_ADDR_OFFSET,
                On_Ctrl4_V2.EXT_COPY_X_ADDR_MASK,
            )
            == ext_x
        )
        assert (
            bit_field(
                ctrl4[0],
                On_Ctrl4_V2.EXT_COPY_Y_ADDR_OFFSET,
                On_Ctrl4_V2.EXT_COPY_Y_ADDR_MASK,
            )
            == ext_y
        )
        assert (
            ctrl4[0]
            == get_frame_dest_v2(FH.CTRL_TYPE4, pkt_offset, pkt_ncopy)
            | expected_ext_payload
        )


@pytest.mark.parametrize(
    "size, weight_width, signed",
    [
        (100, 8, True),
        (128, 8, False),
        (100, 4, True),
        (64, 4, False),
        (60, 2, True),
        (200, 1, True),
        (256, 1, False),
        (0, 8, True),
        (0, 8, False),
    ],
)
def test_weight_uncompressed_unpack(size, weight_width, signed, fixed_rng):
    weight = gen_v2_weight_array(size, weight_width, signed, fixed_rng)
    mapped = weight_dense_pack(weight, weight_width)

    align_size = 128 // weight_width
    expected_size = math.ceil(weight.size / align_size) * (
        128 // (FRAME_DTYPE(0).nbytes * 8)
    )

    assert mapped.dtype == FRAME_DTYPE
    assert mapped.shape == (expected_size,)

    unmapped = weight_dense_unpack(mapped, weight_width, signed, size)
    assert unmapped.dtype == weight.dtype
    assert np.array_equal(weight, unmapped)


@pytest.mark.parametrize(
    "size, weight_width, signed",
    [
        (100, 8, True),
        (128, 8, False),
        (100, 4, True),
        (64, 4, False),
        (60, 2, True),
        (200, 1, True),
        (256, 1, False),
        (0, 8, True),
        (0, 8, False),
    ],
)
def test_weight_csc_unpack(size, weight_width, signed, fixed_rng):
    weight = gen_v2_weight_array(
        size, weight_width, signed, fixed_rng, sparse_ratio=0.4
    )
    mapped = weight_csc_pack(weight, weight_width, input_width=1)

    n_nonzero_per_addr = {1: 7, 2: 7, 4: 6, 8: 5}[weight_width]
    expected_size = math.ceil(np.count_nonzero(weight) / n_nonzero_per_addr) * 2

    assert mapped.dtype == FRAME_DTYPE
    assert mapped.shape == (expected_size,)

    unmapped = weight_csc_unpack(mapped, weight_width, signed, size)
    assert unmapped.dtype == weight.dtype
    assert np.array_equal(weight, unmapped)


def test_weight_csc_pack_8bit_layout():
    weight = np.array([0, 1, 0, 2, 0, 3, 0, 4, 0, 5], dtype=np.uint8)
    mapped = weight_csc_pack(weight, 8, input_width=1)

    expected_low = (
        FRAME_DTYPE(1)
        | (FRAME_DTYPE(2) << 8)
        | (FRAME_DTYPE(3) << 16)
        | (FRAME_DTYPE(4) << 24)
        | (FRAME_DTYPE(5) << 32)
        | (FRAME_DTYPE(1) << 48)
    )
    expected_high = (
        FRAME_DTYPE(3)
        | (FRAME_DTYPE(5) << 16)
        | (FRAME_DTYPE(7) << 32)
        | (FRAME_DTYPE(9) << 48)
    )

    assert np.array_equal(
        mapped, np.array([expected_low, expected_high], dtype=FRAME_DTYPE)
    )


def test_weight_csc_unpack_rejects_odd_frame_count():
    with pytest.raises(ValueError, match="even"):
        weight_csc_unpack(np.array([1], dtype=FRAME_DTYPE), 8, True, 1)


@pytest.mark.parametrize(
    "size, sparse, expectation",
    [
        (64, False, pytest.raises(ValueError)),
        (64, True, contextlib.nullcontext()),
        (65, False, contextlib.nullcontext()),
    ],
    ids=["dense_need_align", "sparse_need_align", "dense_no_align"],
)
def test_weightcsc_unpack_check_aligned(size, sparse, expectation):
    weight = np.ones(size, dtype=np.int8)
    if sparse:
        weight[5] = 0

    with expectation:
        weight_csc_pack(weight, 8, 8)
