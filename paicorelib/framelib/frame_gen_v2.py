from typing import Any

import numpy as np

from ..coordinate import CoordZXYOffset, coordzxy_to_sign_magnitude
from ..core_defs import LCN_EX
from ..core_model_v2 import OfflineCoreRegV2
from ..neuron_model_v2 import (
    OfflineNeuDestInfoV2,
    OfflineNeuFoldedAttrsV2Part1,
    OfflineNeuFoldedAttrsV2Part2,
    OfflineNeuFullAttrsV2Part1,  # Alias of `OfflineNeuHalfAttrsV2`
    OfflineNeuFullAttrsV2Part2,
)
from ..routing_hexa import AERPacketZXYCopy
from .base import FramePackageHeaderV2, FrameV2, WorkFrameV2
from .frame_defs import FrameHeader as FH
from .frame_defs import FramePackageType
from .frame_defs import OfflineConfigFrame1FormatV2 as Off_Cfg1_V2
from .frame_defs import OfflineConfigFrame2FormatV2 as Off_Cfg2_V2
from .frame_defs import OfflineConfigFrame3FormatV2 as Off_Cfg3_V2
from .frame_defs import OfflineControlFrame1FormatV2 as Off_CF1F
from .frame_defs import OfflineControlFrame3FormatV2 as Off_CF3F
from .frame_defs import OfflineWorkFrame1FormatV2 as Off_WF1F_V2
from .types import FRAME_DTYPE, FrameArrayType, LUTActivationType, LUTPotentialType
from .utils import _mask, bin_split

__all__ = ["FrameGenV2", "OfflineFrameGenV2"]


class FrameGenV2:
    @staticmethod
    def make_package(
        header: FH,
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        start_addr: int,
        packages: FrameArrayType,
    ) -> FrameArrayType:
        pkg_header = FramePackageHeaderV2.make_pkg_header(
            header,
            pkt_offset,
            pkt_ncopy,
            start_addr,
            FramePackageType.CONF_TESTOUT,
            len(packages),
        )
        return np.concatenate([pkg_header.value, packages])


class OfflineFrameGenV2(FrameGenV2):
    LCN_TO_TS_AXON_WIDTHS = (
        (8, 9),
        (7, 10),
        (6, 11),
        (5, 12),
        (4, 13),
        (3, 14),
        (2, 15),
        (1, 16),  # LCN_128X
    )

    @staticmethod
    def gen_config_frame1(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        start_addr: int,
        core_reg_: OfflineCoreRegV2 | dict[str, Any],
    ) -> FrameArrayType:
        """Generate a configuration frame type I. The number of packages is calculated automatically."""
        F = Off_Cfg1_V2
        if isinstance(core_reg_, dict):
            core_reg_ = OfflineCoreRegV2.model_validate(core_reg_, strict=True)

        core_reg = core_reg_.model_dump()

        # Convert the test coordzxy in sign-magnitude format
        z, x, y = coordzxy_to_sign_magnitude(
            (core_reg["test_core_xy"], core_reg["test_core_x"], core_reg["test_core_y"])
        )
        test_core_y_h2, test_core_y_l4 = bin_split(y, 4, 2)
        w1 = (
            ((core_reg["snn_ann"] & F.Word1.SNN_ANN_MASK) << F.Word1.SNN_ANN_OFFSET)
            | (
                (core_reg["max_pooling"] & F.Word1.MAX_POOLING_MASK)
                << F.Word1.MAX_POOLING_OFFSET
            )
            | (
                (core_reg["add_potential"] & F.Word1.ADD_POTENTIAL_MASK)
                << F.Word1.ADD_POTENTIAL_OFFSET
            )
            | (
                (core_reg["zero_output"] & F.Word1.ZERO_OUTPUT_MASK)
                << F.Word1.ZERO_OUTPUT_OFFSET
            )
            | (
                (core_reg["input_sign"] & F.Word1.INPUT_SIGN_MASK)
                << F.Word1.INPUT_SIGN_OFFSET
            )
            | (
                (core_reg["input_width"] & F.Word1.INPUT_WIDTH_MASK)
                << F.Word1.INPUT_WIDTH_OFFSET
            )
            | (
                (core_reg["output_sign"] & F.Word1.OUTPUT_SIGN_MASK)
                << F.Word1.OUTPUT_SIGN_OFFSET
            )
            | (
                (core_reg["output_width"] & F.Word1.OUTPUT_WIDTH_MASK)
                << F.Word1.OUTPUT_WIDTH_OFFSET
            )
            | (
                (core_reg["weight_sign"] & F.Word1.WEIGHT_SIGN_MASK)
                << F.Word1.WEIGHT_SIGN_OFFSET
            )
            | (
                (core_reg["weight_width"] & F.Word1.WEIGHT_WIDTH_MASK)
                << F.Word1.WEIGHT_WIDTH_OFFSET
            )
            | ((core_reg["lcn"] & F.Word1.LCN_MASK) << F.Word1.LCN_OFFSET)
            | (
                (core_reg["target_lcn"] & F.Word1.TARGET_LCN_MASK)
                << F.Word1.TARGET_LCN_OFFSET
            )
            | (
                (core_reg["axon_skew"] & F.Word1.AXON_SKEW_MASK)
                << F.Word1.AXON_SKEW_OFFSET
            )
            | (
                (core_reg["neuron_number"] & F.Word1.NEURON_NUMBER_MASK)
                << F.Word1.NEURON_NUMBER_OFFSET
            )
            | ((z & F.Word1.TEST_CORE_XY_MASK) << F.Word1.TEST_CORE_XY_OFFSET)
            | ((x & F.Word1.TEST_CORE_X_MASK) << F.Word1.TEST_CORE_X_OFFSET)
            | (
                (test_core_y_h2 & F.Word1.TEST_CORE_Y_HIGH2_MASK)
                << F.Word1.TEST_CORE_Y_HIGH2_OFFSET
            )
        )
        w2 = (
            (
                (test_core_y_l4 & F.Word2.TEST_CORE_Y_LOW4_MASK)
                << F.Word2.TEST_CORE_Y_LOW4_OFFSET
            )
            | (
                (core_reg["global_send"] & F.Word2.GLOBAL_SEND_MASK)
                << F.Word2.GLOBAL_SEND_OFFSET
            )
            | (
                (core_reg["csc_accelerate"] & F.Word2.CSC_ACCELERATE_MASK)
                << F.Word2.CSC_ACCELERATE_OFFSET
            )
            | (
                (core_reg["global_receive"] & F.Word2.GLOBAL_RECEIVE_MASK)
                << F.Word2.GLOBAL_RECEIVE_OFFSET
            )
            | (
                (core_reg["thread_number"] & F.Word2.THREAD_NUMBER_MASK)
                << F.Word2.THREAD_NUMBER_OFFSET
            )
            | (
                (core_reg["busy_cycle"] & F.Word2.BUSY_CYCLE_MASK)
                << F.Word2.BUSY_CYCLE_OFFSET
            )
            | (
                (core_reg["delay_cycle"] & F.Word2.DELAY_CYCLE_MASK)
                << F.Word2.DELAY_CYCLE_OFFSET
            )
            | (
                (core_reg["width_cycle"] & F.Word2.WIDTH_CYCLE_MASK)
                << F.Word2.WIDTH_CYCLE_OFFSET
            )
        )
        w3 = (
            (
                (core_reg["tick_start"] & F.Word3.TICK_START_MASK)
                << F.Word3.TICK_START_OFFSET
            )
            | (
                (core_reg["tick_duration"] & F.Word3.TICK_DURATION_MASK)
                << F.Word3.TICK_DURATION_OFFSET
            )
            | (
                (core_reg["tick_initial"] & F.Word3.TICK_INITIAL_MASK)
                << F.Word3.TICK_INITIAL_OFFSET
            )
        )

        packages = np.array([w1, w2, w3], dtype=FRAME_DTYPE)
        return OfflineFrameGenV2.make_package(
            FH.CONFIG_TYPE1, pkt_offset, pkt_ncopy, start_addr, packages
        )

    @staticmethod
    def gen_config_frame2(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        arr_pot: LUTPotentialType,
        arr_act: LUTActivationType,
        start_addr: int = 0,
    ) -> FrameArrayType:
        """Generate a configuration frame type II. The number of packages is calculated automatically."""
        F = Off_Cfg2_V2
        N_FRAME_PER_LUT_RAM = 256

        assert len(arr_pot) == len(arr_act)
        packages = np.zeros((N_FRAME_PER_LUT_RAM,), dtype=FRAME_DTYPE)

        arr_pot_u64 = arr_pot.view(np.uint32).astype(FRAME_DTYPE)
        arr_act_u8 = arr_act.view(np.uint8)

        packages = ((arr_pot_u64 << F.POTENTIAL_OFFSET) + arr_act_u8).astype(
            FRAME_DTYPE
        )
        return OfflineFrameGenV2.make_package(
            FH.CONFIG_TYPE2, pkt_offset, pkt_ncopy, start_addr, packages
        )

    @staticmethod
    def gen_config_frame3(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        dest_info: OfflineNeuDestInfoV2 | dict[str, Any],
        full_attrs1: OfflineNeuFullAttrsV2Part1 | dict[str, Any] | None,
        full_attrs2: OfflineNeuFullAttrsV2Part2 | dict[str, Any] | None,
        folded_attrs1: OfflineNeuFoldedAttrsV2Part1 | dict[str, Any] | None,
        folded_attrs2: OfflineNeuFoldedAttrsV2Part2 | dict[str, Any] | None,
        start_addr: int = 0,
        n_package: int = 0,
    ) -> tuple[FramePackageHeaderV2, FrameArrayType, FrameArrayType, FrameArrayType]:
        """Generate the package header, package of half neuron attributes, package of full neuron attributes,   \
            and package of folded neuron attributes. The number of packages can be set later.

        For example:

            >>> header, pkg_half, pkg_full, pkg_fold = OfflineFrameGenV2.gen_config_frame3_component(...)
            >>> header.payload.n_package = 999
        """
        if isinstance(dest_info, dict):
            dest_info = OfflineNeuDestInfoV2.model_validate(dest_info, strict=True)
        if isinstance(full_attrs1, dict):
            full_attrs1 = OfflineNeuFullAttrsV2Part1.model_validate(
                full_attrs1, strict=True
            )
        if isinstance(full_attrs2, dict):
            full_attrs2 = OfflineNeuFullAttrsV2Part2.model_validate(
                full_attrs2, strict=True
            )
        if isinstance(folded_attrs1, dict):
            folded_attrs1 = OfflineNeuFoldedAttrsV2Part1.model_validate(
                folded_attrs1, strict=True
            )
        if isinstance(folded_attrs2, dict):
            folded_attrs2 = OfflineNeuFoldedAttrsV2Part2.model_validate(
                folded_attrs2, strict=True
            )

        header = FramePackageHeaderV2.make_pkg_header(
            FH.CONFIG_TYPE3,
            pkt_offset,
            pkt_ncopy,
            start_addr,
            FramePackageType.CONF_TESTOUT,
            n_package,
        )

        pkg_half_neu = np.array([], dtype=FRAME_DTYPE)
        pkg_full_neu = np.array([], dtype=FRAME_DTYPE)
        pkg_folded_neu = np.array([], dtype=FRAME_DTYPE)

        dest_info_dict = dest_info.model_dump()
        if full_attrs1 is not None:
            full_attrs1_dict = full_attrs1.model_dump()
            # Have a half neuron
            pkg_half_neu = OfflineFrameGenV2._gen_pkg_half_neu(
                dest_info_dict, full_attrs1_dict
            )
            if full_attrs2 is not None:
                # Have a full neuron
                full_attrs2_dict = full_attrs2.model_dump()
                pkg_full_neu = OfflineFrameGenV2._gen_pkg_full_neu(
                    dest_info_dict, full_attrs1_dict, full_attrs2_dict
                )

        if folded_attrs1 is not None and folded_attrs2 is not None:
            folded_attrs1_dict = folded_attrs1.model_dump()
            folded_attrs2_dict = folded_attrs2.model_dump()
            pkg_folded_neu = OfflineFrameGenV2._gen_pkg_folded_neu(
                folded_attrs1_dict, folded_attrs2_dict
            )
        else:
            raise ValueError("attributes of folded neuron are incomplete")

        return header, pkg_half_neu, pkg_full_neu, pkg_folded_neu

    @staticmethod
    def _gen_pkg_half_neu(
        dest_info: dict[str, Any], half_attrs: dict[str, Any]
    ) -> FrameArrayType:
        F = Off_Cfg3_V2.Full

        weight_skew_h11, weight_skew_l5 = bin_split(half_attrs["weight_skew"], 5, 11)
        w1 = (
            (
                (dest_info["tick_relative"] & F.Word1.TICK_RELATIVE_MASK)
                << F.Word1.TICK_RELATIVE_OFFSET
            )
            | (
                (dest_info["addr_axon"] & F.Word1.ADDR_AXON_MASK)
                << F.Word1.ADDR_AXON_OFFSET
            )
            | (
                (dest_info["addr_core_xy"] & F.Word1.ADDR_CORE_XY_MASK)
                << F.Word1.ADDR_CORE_XY_OFFSET
            )
            | (
                (dest_info["addr_core_x"] & F.Word1.ADDR_CORE_X_MASK)
                << F.Word1.ADDR_CORE_X_OFFSET
            )
            | (
                (dest_info["addr_core_y"] & F.Word1.ADDR_CORE_Y_MASK)
                << F.Word1.ADDR_CORE_Y_OFFSET
            )
            | (
                (dest_info["addr_copy_xy"] & F.Word1.ADDR_COPY_XY_MASK)
                << F.Word1.ADDR_COPY_XY_OFFSET
            )
            | (
                (dest_info["addr_copy_x"] & F.Word1.ADDR_COPY_X_MASK)
                << F.Word1.ADDR_COPY_X_OFFSET
            )
            | (
                (dest_info["addr_copy_y"] & F.Word1.ADDR_COPY_Y_MASK)
                << F.Word1.ADDR_COPY_Y_OFFSET
            )
            | (
                (weight_skew_h11 & F.Word1.WEIGHT_SKEW_HIGH11_MASK)
                << F.Word1.WEIGHT_SKEW_HIGH11_OFFSET
            )
        )
        w2 = (
            (
                (weight_skew_l5 & F.Word2.WEIGHT_SKEW_LOW5_MASK)
                << F.Word2.WEIGHT_SKEW_LOW5_OFFSET
            )
            | (
                (half_attrs["weight_address_start"] & F.Word2.WEIGHT_ADDRESS_START_MASK)
                << F.Word2.WEIGHT_ADDRESS_START_OFFSET
            )
            | (
                (half_attrs["weight_address_end"] & F.Word2.WEIGHT_ADDRESS_END_MASK)
                << F.Word2.WEIGHT_ADDRESS_END_OFFSET
            )
            | (
                (half_attrs["output_type"] & F.Word2.OUTPUT_TYPE_MASK)
                << F.Word2.OUTPUT_TYPE_OFFSET
            )
            | (
                (half_attrs["fold_type"] & F.Word2.FOLD_TYPE_MASK)
                << F.Word2.FOLD_TYPE_OFFSET
            )
            | (
                (half_attrs["neuron_type"] & F.Word2.NEURON_TYPE_MASK)
                << F.Word2.NEURON_TYPE_OFFSET
            )
            | ((half_attrs["vjt"] & F.Word2.VJT_MASK) << F.Word2.VJT_OFFSET)
        )

        return np.array([w1, w2], dtype=FRAME_DTYPE)

    @staticmethod
    def _gen_pkg_full_neu(
        dest_info: dict[str, Any],
        full_attrs1: dict[str, Any],
        full_attrs2: dict[str, Any],
    ) -> FrameArrayType:
        F = Off_Cfg3_V2.Full
        pkg_half_neu = OfflineFrameGenV2._gen_pkg_half_neu(dest_info, full_attrs1)
        thres_pos_h12, thres_pos_l20 = bin_split(full_attrs2["threshold_neg"], 20, 12)
        w3 = (
            (
                (full_attrs2["reset_mode"] & F.Word3.RESET_MODE_MASK)
                << F.Word3.RESET_MODE_OFFSET
            )
            | (
                (full_attrs2["reset_v"] & F.Word3.RESET_V_MASK)
                << F.Word3.RESET_V_OFFSET
            )
            | (
                (full_attrs2["threshold_neg_mode"] & F.Word3.THRESHOLD_NEG_MODE_MASK)
                << F.Word3.THRESHOLD_NEG_MODE_OFFSET
            )
            | (
                (full_attrs2["threshold_pos_mode"] & F.Word3.THRESHOLD_POS_MODE_MASK)
                << F.Word3.THRESHOLD_POS_MODE_OFFSET
            )
            | (
                (full_attrs2["threshold_neg"] & F.Word3.THRESHOLD_NEG_MASK)
                << F.Word3.THRESHOLD_NEG_OFFSET
            )
            | (
                (thres_pos_h12 & F.Word3.THRESHOLD_POS_HIGH12_MASK)
                << F.Word3.THRESHOLD_POS_HIGH12_OFFSET
            )
        )
        w4 = (
            (
                (thres_pos_l20 & F.Word4.THRESHOLD_POS_LOW20_MASK)
                << F.Word4.THRESHOLD_POS_LOW20_OFFSET
            )
            | (
                (full_attrs2["lateral_inhibit"] & F.Word4.LATERAL_INHIBITION_MASK)
                << F.Word4.LATERAL_INHIBITION_OFFSET
            )
            | (
                (full_attrs2["leak_multi_seq"] & F.Word4.LEAK_MULTI_SEQUENCE_MASK)
                << F.Word4.LEAK_MULTI_SEQUENCE_OFFSET
            )
            | (
                (full_attrs2["leak_multi_in"] & F.Word4.LEAK_MULTI_INPUT_MASK)
                << F.Word4.LEAK_MULTI_INPUT_OFFSET
            )
            | (
                (full_attrs2["leak_multi_mode"] & F.Word4.LEAK_MULTI_MODE_MASK)
                << F.Word4.LEAK_MULTI_MODE_OFFSET
            )
            | (
                (full_attrs2["leak_add_mode"] & F.Word4.LEAK_ADD_MODE_MASK)
                << F.Word4.LEAK_ADD_MODE_OFFSET
            )
            | (
                (full_attrs2["leak_tau"] & F.Word4.LEAK_TAU_MASK)
                << F.Word4.LEAK_TAU_OFFSET
            )
            | ((full_attrs2["leak_v"] & F.Word4.LEAK_V_MASK) << F.Word4.LEAK_V_OFFSET)
            | (
                (full_attrs2["weight_compress"] & F.Word4.WEIGHT_COMPRESS_MASK)
                << F.Word4.WEIGHT_COMPRESS_OFFSET
            )
            | (
                (full_attrs2["vjt_initial"] & F.Word4.VJT_INITIAL_MASK)
                << F.Word4.VJT_INITIAL_OFFSET
            )
        )

        pkg_full_neu_part2 = np.array([w3, w4], dtype=FRAME_DTYPE)
        return np.concatenate([pkg_half_neu, pkg_full_neu_part2])

    @staticmethod
    def _gen_pkg_folded_neu(
        folded_attrs1: dict[str, Any], folded_attrs2: dict[str, Any]
    ) -> FrameArrayType:
        F = Off_Cfg3_V2.Fold
        fold_skew_y_h9, fold_skew_y_l5 = bin_split(folded_attrs2["fold_skew_y"], 5, 9)
        w1 = (
            (
                (folded_attrs1["fold_range_xy"] & F.Word1.FOLD_RANGE_XY_MASK)
                << F.Word1.FOLD_RANGE_XY_OFFSET
            )
            | (
                (folded_attrs1["fold_range_x"] & F.Word1.FOLD_RANGE_X_MASK)
                << F.Word1.FOLD_RANGE_X_OFFSET
            )
            | (
                (folded_attrs1["fold_range_y"] & F.Word1.FOLD_RANGE_Y_MASK)
                << F.Word1.FOLD_RANGE_Y_OFFSET
            )
            | (
                (folded_attrs1["fold_skew_xy"] & F.Word1.FOLD_SKEW_XY_MASK)
                << F.Word1.FOLD_SKEW_XY_OFFSET
            )
            | (
                (folded_attrs1["fold_skew_x"] & F.Word1.FOLD_SKEW_X_MASK)
                << F.Word1.FOLD_SKEW_X_OFFSET
            )
            | (
                (fold_skew_y_h9 & F.Word1.FOLD_SKEW_Y_HIGH9_MASK)
                << F.Word1.FOLD_SKEW_Y_HIGH9_OFFSET
            )
        )
        w2 = (
            (
                (fold_skew_y_l5 & F.Word2.FOLD_SKEW_Y_LOW2_MASK)
                << F.Word2.FOLD_SKEW_Y_LOW2_OFFSET
            )
            | (
                (folded_attrs1["fold_axon_xy"] & F.Word2.FOLD_AXON_XY_MASK)
                << F.Word2.FOLD_AXON_XY_OFFSET
            )
            | (
                (folded_attrs1["fold_axon_x"] & F.Word2.FOLD_AXON_X_MASK)
                << F.Word2.FOLD_AXON_X_OFFSET
            )
            | (
                (folded_attrs1["fold_axon_y"] & F.Word2.FOLD_AXON_Y_MASK)
                << F.Word2.FOLD_AXON_Y_OFFSET
            )
            | (
                (folded_attrs1["fold_number"] & F.Word2.FOLD_NUMBER_MASK)
                << F.Word2.FOLD_NUMBER_OFFSET
            )
        )
        w3 = (
            (folded_attrs2["fold_vjt_3"] & F.Word3.FOLD_VJT_3_MASK)
            << F.Word3.FOLD_VJT_3_OFFSET
        ) | (
            (folded_attrs2["fold_vjt_2"] & F.Word3.FOLD_VJT_2_MASK)
            << F.Word3.FOLD_VJT_2_OFFSET
        )
        w4 = (
            (folded_attrs2["fold_vjt_1"] & F.Word4.FOLD_VJT_1_MASK)
            << F.Word4.FOLD_VJT_1_OFFSET
        ) | (
            (folded_attrs2["fold_vjt_0"] & F.Word4.FOLD_VJT_0_MASK)
            << F.Word4.FOLD_VJT_0_OFFSET
        )

        return np.array([w1, w2, w3, w4], dtype=FRAME_DTYPE)

    # @staticmethod
    # def gen_test_request(
    #     header: FH,
    #     pkt_offset: CoordZXYOffset,
    #     pkt_ncopy: AERPacketZXYCopy,
    #     pkg_type: FramePackageType,
    #     start_addr: int,
    #     n_package: int,
    # ) -> FrameArrayType:
    #     return OfflineFrameGenV2.make_pkg_header(
    #         header, pkt_offset, pkt_ncopy, pkg_type, start_addr, n_package
    #     )

    @staticmethod
    def gen_work_frame1(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        timestep: int,
        axon: int,
        data: int,
        target_lcn: LCN_EX,
    ) -> FrameArrayType:
        F = Off_WF1F_V2
        ts_wdith, axon_width = OfflineFrameGenV2.LCN_TO_TS_AXON_WIDTHS[target_lcn.value]

        ts_msb = (timestep >> (ts_wdith - 1)) & F.TIMESTEP_HIGH7_MASK
        ts_low = timestep & _mask(ts_wdith - 1)

        # Payload = ts[MSB-1:0] + axon + data
        payload = (
            (ts_low << (F.AXON_ADDR_OFFSET + axon_width))
            | ((axon & _mask(axon_width)) << F.AXON_ADDR_OFFSET)
            | ((data & F.DATA_MASK) << F.DATA_OFFSET)
        )

        return WorkFrameV2(FH.WORK_TYPE1, pkt_offset, pkt_ncopy, payload, ts_msb).value

    @staticmethod
    def gen_control_frame1(
        pkt_offset: CoordZXYOffset, pkt_ncopy: AERPacketZXYCopy, n_timestep: int
    ) -> FrameArrayType:
        F = Off_CF1F
        if n_timestep > F.NUM_TIMESTEP_MASK:
            raise ValueError(f"'overflow' out of range {F.NUM_TIMESTEP_MASK}")

        return FrameV2(FH.CTRL_TYPE1, pkt_offset, pkt_ncopy, n_timestep).value

    @staticmethod
    def gen_control_frame2(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
    ) -> FrameArrayType:
        return FrameV2(FH.CTRL_TYPE2, pkt_offset, pkt_ncopy, 0).value

    @staticmethod
    def gen_complete_frame(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        thread_id: int = 0,
    ) -> FrameArrayType:
        F = Off_CF3F
        if thread_id > F.THREAD_ID_MASK:
            raise ValueError(f"'thread_id' out of range {F.THREAD_ID_MASK})")

        return FrameV2(FH.CTRL_TYPE3, pkt_offset, pkt_ncopy, thread_id).value


# class OfflineFrameDecoderV2:
#     @staticmethod
#     def decode_voltage(frame: FrameArrayType, target_lcn: LCN_EX) -> FrameArrayType:
#         pass
