import contextlib
import math
from typing import Literal, cast

import numpy as np
import pytest

from paicorelib.coordinate import coordzxy_to_sign_magnitude
from paicorelib.core_defs import LCN_EX
from paicorelib.core_defs_v2 import CSCAccelerateMode, DataSign, DataWidth
from paicorelib.core_model_v2 import OfflineCoreRegV2
from paicorelib.framelib import FRAME_DTYPE
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
    OfflineControlFrame1FormatV2 as Off_Ctrl1_V2,
)
from paicorelib.framelib.frame_defs import (
    OfflineControlFrame3FormatV2 as Off_Ctrl3_V2,
)
from paicorelib.framelib.frame_defs import (
    OfflineWorkFrame1FormatV2 as Off_Work1_V2,
)
from paicorelib.framelib.frame_defs import OnlineConfigFrame1FormatV2 as On_Cfg1_V2
from paicorelib.framelib.frame_defs import OnlineConfigFrame2FormatV2 as On_Cfg2_V2
from paicorelib.framelib.frame_defs import OnlineConfigFrame3FormatV2 as On_Cfg3_V2
from paicorelib.framelib.frame_defs import OnlineControlFrame1FormatV2 as On_Ctrl1_V2
from paicorelib.framelib.frame_defs import OnlineControlFrame3FormatV2 as On_Ctrl3_V2
from paicorelib.framelib.frame_defs import OnlineControlFrame4FormatV2 as On_Ctrl4_V2
from paicorelib.framelib.frame_defs import OnlineWorkFrame1FormatV2 as On_Work1_V2
from paicorelib.framelib.frame_gen_v2 import (
    DataWidthLE8Like,
    FrameGenV2,
    OfflineFrameGenV2,
    OnlineFrameGenV2,
    weight_csc_pack,
    weight_csc_u16_pack,
    weight_csc_unpack,
    weight_dense_pack,
    weight_dense_u16_pack,
    weight_dense_unpack,
)
from paicorelib.framelib.types import (
    LUT_ACTIVATION_DTYPE,
    LUT_POTENTIAL_DTYPE,
    FrameArrayType,
)
from paicorelib.framelib.utils import single_frame_header_check
from paicorelib.neuron_defs import ResetMode
from paicorelib.neuron_defs_v2 import (
    FoldType,
    LateralInhibitionMode,
    LeakAddMode,
    LeakMultiComparisonOrder,
    LeakMultiInputMode,
    LeakMultiMode,
    NeuronType,
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
    OnlineNeuFoldedAttrsV2Part1,
    OnlineNeuFoldedAttrsV2Part2,
)
from paicorelib.utils import _mask
from tests.utils import (
    bit_field,
    build_v2_core_reg_params,
    build_v2_dest_info_params,
    build_v2_folded_attrs_part1_params,
    build_v2_folded_attrs_part2_params,
    build_v2_full_attrs_part2_params,
    build_v2_half_attrs_params,
    build_v2_weight_array,
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


def scalar_bits(value, dtype) -> int:
    arr = np.asarray([value], dtype=dtype)
    view_dtype = {2: np.uint16, 4: np.uint32, 8: np.uint64}[arr.dtype.itemsize]
    return int(arr.view(view_dtype)[0])


def scalar_bf16_bits(value) -> int:
    arr = np.asarray([value], dtype=np.dtype(np.float32).newbyteorder("<"))
    bits = np.ascontiguousarray(arr).view(np.dtype(np.uint32).newbyteorder("<"))
    return int(bits[0] >> 16)


def array_bf16_bits(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.dtype(np.float32).newbyteorder("<"))
    bits = np.ascontiguousarray(arr).view(np.dtype(np.uint32).newbyteorder("<"))
    return (bits >> 16).astype(np.uint16)


def bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    arr_u32 = np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16
    return arr_u32.view(np.float32)


def build_online_v2_core_reg_params(**overrides):
    base = {
        "snn_ann": 1,
        "max_pooling": 0,
        "add_potential": 1,
        "zero_output": 0,
        "work_mode": 5,
        "input_core": 1,
        "input_width": 2,
        "output_core": 1,
        "output_width": 3,
        "LCN_AT": 1,
        "LCN_MP": 2,
        "LCN_LG": 3,
        "target_LCN_AT": 4,
        "target_LCN_MP": 5,
        "target_LCN_LG": 6,
        "axon_skew": 0x1234,
        "neuron_number": 0x1456,
        "update_number": 0x1555,
        "csc_accelerate": CSCAccelerateMode.ENABLE,
        "scale_in": np.float32(1.5),
        "bias_in": np.float32(-2.0),
        "scale_out": np.float32(0.25),
        "bias_out": np.float32(-0.5),
        "learning_rate": np.float32(3.0),
        "update_core_xy": -1,
        "update_core_x": 2,
        "update_core_y": -3,
        "test_core_xy": 4,
        "test_core_x": -5,
        "test_core_y": -6,
        "global_send": 0x55,
        "global_receive": 0x2A,
        "thread_number": 0x155,
        "busy_cycle": 0xABC,
        "delay_cycle": 0x1234,
        "width_cycle": 0x56,
        "tick_start": 0x789A,
        "tick_duration": 0x12345678,
        "tick_initial": 0x9ABC,
    }
    base.update(overrides)
    return base


def normalize_width_bits(value: DataWidthLE8Like) -> WidthBitsLE8:
    width_bits = (1 << value.value) if isinstance(value, DataWidth) else int(value)
    return cast(WidthBitsLE8, width_bits)


def validated_dump(model_cls, params):
    return model_cls.model_validate(params, strict=True).model_dump()


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

        frames = OfflineFrameGenV2.gen_config_frame2(pkt_offset, arr_pot, arr_act)

        assert frames.size == 257

        arr_pot2, arr_act2 = extract_lut_from_cf2(frames, act_dtype)
        assert np.array_equal(arr_pot, arr_pot2)
        assert np.array_equal(arr_act, arr_act2)

    def test_cf2_rejects_mismatched_lut_size(self, v2_packet_route):
        pkt_offset, _ = v2_packet_route
        arr_pot = np.arange(255, dtype=LUT_POTENTIAL_DTYPE)
        arr_act = np.arange(256, dtype=np.uint8)

        with pytest.raises(ValueError, match="same size"):
            OfflineFrameGenV2.gen_config_frame2(pkt_offset, arr_pot, arr_act)

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
        ) == (dest_info["addr_core_y"] & Off_Cfg3_V2.Full.Word2.ADDR_CORE_Y_MASK)

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

    def test_gen_config_frame3_pkg_folded_wrapper_rejects_incomplete_attrs(self):
        with pytest.raises(ValueError, match="incomplete"):
            OfflineFrameGenV2.gen_config_frame3_pkg_folded(
                build_v2_folded_attrs_part1_params(), []
            )

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
        weight = build_v2_weight_array(
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

    @pytest.mark.parametrize(
        "timesteps, axons, data",
        [
            ([1, 2], [3], [4, 5]),
            ([1, 2], [3, 4], [5]),
        ],
    )
    def test_wf1_rejects_mismatched_lengths(
        self, v2_packet_route, timesteps, axons, data
    ):
        pkt_offset, pkt_ncopy = v2_packet_route

        with pytest.raises(ValueError, match="size"):
            OfflineFrameGenV2.gen_work_frame1(
                pkt_offset, pkt_ncopy, timesteps, axons, LCN_EX.LCN_1X, data
            )

    def test_control_frames(self, v2_packet_route):
        pkt_offset, pkt_ncopy = v2_packet_route
        ctrl1 = OfflineFrameGenV2.gen_control_frame1(pkt_offset, pkt_ncopy, 123)
        ctrl2 = OfflineFrameGenV2.gen_control_frame2(pkt_offset, pkt_ncopy)
        complete = OfflineFrameGenV2.gen_complete_frame(pkt_offset, pkt_ncopy, 7)

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

    def test_control_frame1_rejects_overflow(self, v2_packet_route):
        pkt_offset, pkt_ncopy = v2_packet_route
        with pytest.raises(ValueError, match="overflow"):
            OfflineFrameGenV2.gen_control_frame1(
                pkt_offset, pkt_ncopy, Off_Ctrl1_V2.NUM_TIMESTEP_MASK + 1
            )

    def test_complete_frame_rejects_overflow(self, v2_packet_route):
        pkt_offset, pkt_ncopy = v2_packet_route
        with pytest.raises(ValueError, match="thread_id"):
            OfflineFrameGenV2.gen_complete_frame(
                pkt_offset, pkt_ncopy, Off_Ctrl3_V2.THREAD_ID_MASK + 1
            )


class TestOnlineFrameGenV2:
    def test_cf1(self, v2_packet_route):
        pkt_offset, pkt_ncopy = v2_packet_route
        core_reg = build_online_v2_core_reg_params()

        frames = OnlineFrameGenV2.gen_config_frame1(pkt_offset, core_reg, pkt_ncopy, 0)
        update_core_xy, update_core_x, update_core_y = coordzxy_to_sign_magnitude(
            (
                core_reg["update_core_xy"],
                core_reg["update_core_x"],
                core_reg["update_core_y"],
            )
        )
        test_core_xy, test_core_x, test_core_y_expected = coordzxy_to_sign_magnitude(
            (
                core_reg["test_core_xy"],
                core_reg["test_core_x"],
                core_reg["test_core_y"],
            )
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
            == core_reg["work_mode"]
        )
        assert neuron_number == core_reg["neuron_number"]
        assert scale_out_bits == scalar_bf16_bits(core_reg["scale_out"])
        assert bit_field(
            frames[2],
            On_Cfg1_V2.Word2.SCALE_IN_OFFSET,
            On_Cfg1_V2.Word2.SCALE_IN_MASK,
        ) == scalar_bf16_bits(core_reg["scale_in"])
        assert bit_field(
            frames[3],
            On_Cfg1_V2.Word3.BIAS_OUT_OFFSET,
            On_Cfg1_V2.Word3.BIAS_OUT_MASK,
        ) == scalar_bf16_bits(core_reg["bias_out"])
        assert bit_field(
            frames[3],
            On_Cfg1_V2.Word3.LEARNING_RATE_OFFSET,
            On_Cfg1_V2.Word3.LEARNING_RATE_MASK,
        ) == scalar_bf16_bits(core_reg["learning_rate"])
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
            == core_reg["tick_duration"]
        )

    def test_cf1_rejects_invalid_core_params(self, v2_packet_route):
        pkt_offset, pkt_ncopy = v2_packet_route

        with pytest.raises(ValueError, match="update_core_xy"):
            OnlineFrameGenV2.gen_config_frame1(
                pkt_offset,
                build_online_v2_core_reg_params(update_core_xy=32),
                pkt_ncopy,
                0,
            )

    @pytest.mark.parametrize("act_dtype", [np.int16, np.float32])
    def test_cf2(self, v2_packet_route, act_dtype):
        pkt_offset, _ = v2_packet_route
        arr_pot = np.arange(-128, 128, dtype=np.int32)
        if act_dtype == np.float32:
            arr_act = (np.arange(256, dtype=np.float32) / 16) - np.float32(8.0)
        else:
            arr_act = np.arange(-128, 128, dtype=act_dtype)

        frames = OnlineFrameGenV2.gen_config_frame2(pkt_offset, arr_pot, arr_act)

        assert frames.size == 257

        arr_pot2, arr_act_bits = extract_online_lut_from_cf2(frames)
        assert np.array_equal(arr_pot, arr_pot2)
        if act_dtype == np.float32:
            expected_act_bits = array_bf16_bits(arr_act)
            assert np.array_equal(arr_act_bits, expected_act_bits)
            assert np.array_equal(
                bf16_bits_to_float32(arr_act_bits),
                bf16_bits_to_float32(expected_act_bits),
            )
        else:
            assert np.array_equal(arr_act_bits.view(act_dtype), arr_act)

    def test_cf2_rejects_mismatched_lut_size(self, v2_packet_route):
        pkt_offset, _ = v2_packet_route
        arr_pot = np.arange(255, dtype=np.int32)
        arr_act = np.arange(256, dtype=np.int16)

        with pytest.raises(ValueError, match="same size"):
            OnlineFrameGenV2.gen_config_frame2(pkt_offset, arr_pot, arr_act)

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
        half_attrs = build_v2_half_attrs_params(
            weight_skew=0xABC,
            weight_address_start=0x123,
            weight_address_end=0x456,
            output_type=OutputType.POTENTIAL,
            fold_type=FoldType.UNFOLDED,
            neuron_type=NeuronType.FULL,
            vjt=np.float32(1.25),
        )

        pkg_half_neu = OnlineFrameGenV2._gen_pkg_half_neu(dest_info, half_attrs)
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
        ) == scalar_bits(half_attrs["vjt"], np.float32)
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

    def test_gen_pkg_half_neu_rejects_invalid_dest_info(self):
        with pytest.raises(ValueError, match="addr_axon"):
            OnlineFrameGenV2.gen_config_frame3_pkg_half(
                build_v2_dest_info_params(addr_axon=0x1FF),
                build_v2_half_attrs_params(vjt=np.float32(0.5)),
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
        full_attrs1 = build_v2_half_attrs_params(
            weight_skew=0x123,
            weight_address_start=10,
            weight_address_end=20,
            output_type=OutputType.POTENTIAL,
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

        pkg_full_neu = OnlineFrameGenV2._gen_pkg_full_neu(
            dest_info, full_attrs1, full_attrs2
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
        assert threshold_pos_bits == scalar_bits(
            full_attrs2["threshold_pos"], np.float32
        )
        assert bit_field(
            pkg_full_neu[3],
            On_Cfg3_V2.Full.Word4.THRESHOLD_NEG_OFFSET,
            On_Cfg3_V2.Full.Word4.THRESHOLD_NEG_MASK,
        ) == scalar_bits(full_attrs2["threshold_neg"], np.float32)
        assert bit_field(
            pkg_full_neu[3],
            On_Cfg3_V2.Full.Word4.RESET_V_OFFSET,
            On_Cfg3_V2.Full.Word4.RESET_V_MASK,
        ) == scalar_bf16_bits(full_attrs2["reset_v"])
        assert bit_field(
            pkg_full_neu[2],
            On_Cfg3_V2.Full.Word3.LEAK_V_OFFSET,
            On_Cfg3_V2.Full.Word3.LEAK_V_MASK,
        ) == scalar_bf16_bits(full_attrs2["leak_v"])
        assert bit_field(
            pkg_full_neu[2],
            On_Cfg3_V2.Full.Word3.VJT_INITIAL_OFFSET,
            On_Cfg3_V2.Full.Word3.VJT_INITIAL_MASK,
        ) == scalar_bf16_bits(full_attrs2["vjt_initial"])
        assert (
            bit_field(
                pkg_full_neu[2],
                On_Cfg3_V2.Full.Word3.WEIGHT_COMPRESS_OFFSET,
                On_Cfg3_V2.Full.Word3.WEIGHT_COMPRESS_MASK,
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
        ) == scalar_bits(attrs2_1["fold_vjt_0"], np.float32)
        assert bit_field(
            pkg_folded_neu[3],
            On_Cfg3_V2.Fold.Word4.FOLD_VJT_3_OFFSET,
            On_Cfg3_V2.Fold.Word4.FOLD_VJT_3_MASK,
        ) == scalar_bits(attrs2_1["fold_vjt_3"], np.float32)
        assert bit_field(
            pkg_folded_neu[4],
            On_Cfg3_V2.Fold.Word3.FOLD_VJT_1_OFFSET,
            On_Cfg3_V2.Fold.Word3.FOLD_VJT_1_MASK,
        ) == scalar_bits(attrs2_2["fold_vjt_1"], np.float32)
        assert bit_field(
            pkg_folded_neu[5],
            On_Cfg3_V2.Fold.Word4.FOLD_VJT_2_OFFSET,
            On_Cfg3_V2.Fold.Word4.FOLD_VJT_2_MASK,
        ) == scalar_bits(attrs2_2["fold_vjt_2"], np.float32)

    @pytest.mark.parametrize(
        "wrapper_name,args,expected_size",
        [
            (
                "gen_config_frame3_pkg_half",
                (
                    build_v2_dest_info_params(),
                    build_v2_half_attrs_params(vjt=np.float32(0.5)),
                ),
                2,
            ),
            (
                "gen_config_frame3_pkg_full",
                (
                    build_v2_dest_info_params(),
                    build_v2_half_attrs_params(vjt=np.float32(0.5)),
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

    def test_gen_config_frame3_pkg_folded_wrapper_rejects_incomplete_attrs(self):
        with pytest.raises(ValueError, match="incomplete"):
            OnlineFrameGenV2.gen_config_frame3_pkg_folded(
                build_v2_folded_attrs_part1_params(), []
            )

    def test_gen_config_frame3_pkg_neu_valid_combinations(self):
        result = OnlineFrameGenV2.gen_config_frame3_pkg_neu(
            build_v2_dest_info_params(),
            build_v2_half_attrs_params(vjt=np.float32(0.5)),
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
                    build_v2_half_attrs_params(),
                    None,
                    build_v2_folded_attrs_part1_params(),
                    [],
                ),
                "folded neuron.*missing part2",
            ),
            (
                (
                    build_v2_dest_info_params(),
                    build_v2_half_attrs_params(),
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
            (np.array([1, -2, 3, 4, 5], dtype=np.int16), False, weight_dense_u16_pack),
            (
                np.array([0, 1, 0, -2, 0, 3, 0, 4], dtype=np.int16),
                CSCAccelerateMode.ENABLE,
                weight_csc_u16_pack,
            ),
        ],
        ids=["dense", "csc"],
    )
    def test_gen_config_frame3_weight_pkg(self, weight, csc_compress, expected_builder):
        result = OnlineFrameGenV2.gen_config_frame3_weight_pkg(weight, csc_compress)
        expected = expected_builder(weight)

        assert np.array_equal(result, expected)

    def test_gen_config_frame3_weight_pkg_dense_bf16_float(self):
        weight = np.array([1.5, -2.0, 3.25], dtype=np.float32)
        result = OnlineFrameGenV2.gen_config_frame3_weight_pkg(weight, False)
        expected_bits = array_bf16_bits(weight)

        assert result.shape == (2,)
        assert (
            bit_field(
                result[0],
                On_Cfg3_V2.WeightDense.Word1.WEIGHT_0_OFFSET,
                On_Cfg3_V2.WeightDense.Word1.WEIGHT_0_MASK,
            )
            == expected_bits[0]
        )
        assert (
            bit_field(
                result[0],
                On_Cfg3_V2.WeightDense.Word1.WEIGHT_1_OFFSET,
                On_Cfg3_V2.WeightDense.Word1.WEIGHT_1_MASK,
            )
            == expected_bits[1]
        )
        assert (
            bit_field(
                result[0],
                On_Cfg3_V2.WeightDense.Word1.WEIGHT_2_OFFSET,
                On_Cfg3_V2.WeightDense.Word1.WEIGHT_2_MASK,
            )
            == expected_bits[2]
        )
        assert (
            bit_field(
                result[0],
                On_Cfg3_V2.WeightDense.Word1.WEIGHT_3_OFFSET,
                On_Cfg3_V2.WeightDense.Word1.WEIGHT_3_MASK,
            )
            == 0
        )

    def test_gen_config_frame3_weight_pkg_rejects_non_1d_weight(self):
        with pytest.raises(ValueError, match="1D"):
            OnlineFrameGenV2.gen_config_frame3_weight_pkg(
                np.ones((2, 2), dtype=np.int16)
            )

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
        "method_name,header,data,expected_bytes",
        [
            (
                "gen_work_frame1",
                FH.WORK_TYPE1,
                np.array([0x1234, 0], dtype=np.uint16),
                np.array([0x34, 0x12], dtype=np.uint8),
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
        ],
        ids=["wf1-u16", "wf2-f16", "wf3-f32"],
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

        assert wf.dtype == FRAME_DTYPE
        assert wf.size == expected_bytes.size
        assert all(single_frame_header_check(frame, header) for frame in wf)
        assert np.array_equal(
            (wf & On_Work1_V2.DATA_MASK).astype(np.uint8), expected_bytes
        )
        assert np.all(
            ((wf >> On_Work1_V2.TIMESTEP_AXON_OFFSET) & On_Work1_V2.TIMESTEP_AXON_MASK)
            == ((3 << 8) | 5)
        )

    def test_work_frame4_supports_target_lcn_8(self, v2_packet_route):
        pkt_offset, pkt_ncopy = v2_packet_route
        wf = OnlineFrameGenV2.gen_work_frame4(
            pkt_offset,
            pkt_ncopy,
            np.array([7]),
            np.array([0xBEEF]),
            8,
            np.array([1], dtype=np.uint8),
        )

        assert wf.shape == (1,)
        assert (
            bit_field(
                wf[0], On_Work1_V2.TIMESTEP_AXON_OFFSET, On_Work1_V2.TIMESTEP_AXON_MASK
            )
            == 0xBEEF
        )

    @pytest.mark.parametrize(
        "timesteps, axons, data",
        [
            ([1, 2], [3], [4, 5]),
            ([1, 2], [3, 4], [5]),
        ],
    )
    def test_work_frame_rejects_mismatched_lengths(
        self, v2_packet_route, timesteps, axons, data
    ):
        pkt_offset, pkt_ncopy = v2_packet_route

        with pytest.raises(ValueError, match="size"):
            OnlineFrameGenV2.gen_work_frame1(
                pkt_offset, pkt_ncopy, timesteps, axons, LCN_EX.LCN_1X, data
            )

    def test_control_frames(self, v2_packet_route):
        pkt_offset, pkt_ncopy = v2_packet_route
        ctrl1 = OnlineFrameGenV2.gen_control_frame1(pkt_offset, pkt_ncopy, 123)
        ctrl2 = OnlineFrameGenV2.gen_control_frame2(pkt_offset, pkt_ncopy)
        complete = OnlineFrameGenV2.gen_complete_frame(pkt_offset, pkt_ncopy, 7)
        update = OnlineFrameGenV2.gen_update_frame(pkt_offset, pkt_ncopy, 0x12345)

        assert single_frame_header_check(ctrl1[0], FH.CTRL_TYPE1)
        assert single_frame_header_check(ctrl2[0], FH.CTRL_TYPE2)
        assert single_frame_header_check(complete[0], FH.CTRL_TYPE3)
        assert single_frame_header_check(update[0], FH.CTRL_TYPE4)
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
        assert (
            bit_field(update[0], FFV2.GENERAL_PAYLOAD_OFFSET, FFV2.GENERAL_PAYLOAD_MASK)
            == 0x12345
        )

    def test_control_frame1_rejects_overflow(self, v2_packet_route):
        pkt_offset, pkt_ncopy = v2_packet_route
        with pytest.raises(ValueError, match="overflow"):
            OnlineFrameGenV2.gen_control_frame1(
                pkt_offset, pkt_ncopy, On_Ctrl1_V2.NUM_TIMESTEP_MASK + 1
            )

    def test_complete_frame_rejects_overflow(self, v2_packet_route):
        pkt_offset, pkt_ncopy = v2_packet_route
        with pytest.raises(ValueError, match="thread_id"):
            OnlineFrameGenV2.gen_complete_frame(
                pkt_offset, pkt_ncopy, On_Ctrl3_V2.THREAD_ID_MASK + 1
            )

    def test_update_frame_rejects_overflow(self, v2_packet_route):
        pkt_offset, pkt_ncopy = v2_packet_route
        with pytest.raises(ValueError, match="ext_multicast_addr"):
            OnlineFrameGenV2.gen_update_frame(
                pkt_offset, pkt_ncopy, On_Ctrl4_V2.EXT_MULTICAST_ADDR_MASK + 1
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
    weight = build_v2_weight_array(size, weight_width, signed, fixed_rng)
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
    weight = build_v2_weight_array(
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
