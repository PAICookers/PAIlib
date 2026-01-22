import math
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from paicorelib.framelib.frames import NDArray

from ..coordinate import CoordZXYOffset, coordzxy_to_sign_magnitude
from ..core_defs import LCN_EX
from ..core_defs_v2 import CSCAccelerateMode, DataWidth
from ..core_model_v2 import OfflineCoreRegV2
from ..neuron_model_v2 import (
    OfflineNeuDestInfoV2,
    OfflineNeuFoldedAttrsV2Part1,
    OfflineNeuFoldedAttrsV2Part2,
    OfflineNeuFullAttrsV2Part1,
    OfflineNeuFullAttrsV2Part2,
)
from ..routing_hexa import AERPacketZXYCopy
from .base import FramePackageHeaderV2, FrameV2, get_frame_destV2
from .frame_defs import FrameHeader as FH
from .frame_defs import FramePackageType
from .frame_defs import OfflineConfigFrame1FormatV2 as Off_Cfg1_V2
from .frame_defs import OfflineConfigFrame2FormatV2 as Off_Cfg2_V2
from .frame_defs import OfflineConfigFrame3FormatV2 as Off_Cfg3_V2
from .frame_defs import OfflineControlFrame1FormatV2 as Off_Ctrl1_V2
from .frame_defs import OfflineControlFrame3FormatV2 as Off_Ctrl3_V2
from .frame_defs import OfflineWorkFrame1FormatV2 as Off_Work1_V2
from .types import (
    FRAME_DTYPE,
    PAYLOAD_DATA_DTYPE,
    FrameArrayType,
    LUTActivationType,
    LUTPotentialType,
)
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
        core_reg_: OfflineCoreRegV2 | dict[str, Any],
        pkt_ncopy: AERPacketZXYCopy = AERPacketZXYCopy(),
        start_addr: int = 0,
    ) -> FrameArrayType:
        """Generate a configuration frame type I. The number of packages is calculated automatically."""
        F = Off_Cfg1_V2
        core_reg = OfflineCoreRegV2.model_validate(core_reg_, strict=True).model_dump()

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

        pkg = np.array([w1, w2, w3], dtype=FRAME_DTYPE)
        return OfflineFrameGenV2.make_package(
            FH.CONFIG_TYPE1, pkt_offset, pkt_ncopy, start_addr, pkg
        )

    @staticmethod
    def gen_config_frame2(
        pkt_offset: CoordZXYOffset,
        potentials: LUTPotentialType,
        activations: LUTActivationType,
        pkt_ncopy: AERPacketZXYCopy = AERPacketZXYCopy(),
        start_addr: int = 0,
    ) -> FrameArrayType:
        """Generate a configuration frame type II. The number of packages is calculated automatically."""
        F = Off_Cfg2_V2
        N_FRAME_PER_LUT_RAM = 256

        assert potentials.size == activations.size
        packages = np.zeros((N_FRAME_PER_LUT_RAM,), dtype=FRAME_DTYPE)

        arr_pot_u64 = potentials.view(np.uint32).astype(FRAME_DTYPE)
        arr_act_u8 = activations.astype(np.uint8)

        packages = ((arr_pot_u64 << F.POTENTIAL_OFFSET) + arr_act_u8).astype(
            FRAME_DTYPE
        )
        return OfflineFrameGenV2.make_package(
            FH.CONFIG_TYPE2, pkt_offset, pkt_ncopy, start_addr, packages
        )

    @staticmethod
    def gen_config_frame3_pkg_header(
        pkt_offset: CoordZXYOffset,
        start_addr: int,
        n_package: int,
        pkt_ncopy: AERPacketZXYCopy = AERPacketZXYCopy(),
    ) -> FrameArrayType:
        """Generate the package header of configuration frame type III."""
        header = FramePackageHeaderV2.make_pkg_header(
            FH.CONFIG_TYPE3,
            pkt_offset,
            pkt_ncopy,
            start_addr,
            FramePackageType.CONF_TESTOUT,
            n_package,
        )
        return header.value

    @staticmethod
    def gen_config_frame3_pkg_neu(
        dest_info: OfflineNeuDestInfoV2 | dict[str, Any],
        full_attrs1: OfflineNeuFullAttrsV2Part1 | dict[str, Any] | None,
        full_attrs2: OfflineNeuFullAttrsV2Part2 | dict[str, Any] | None,
        folded_attrs1: OfflineNeuFoldedAttrsV2Part1 | dict[str, Any] | None,
        folded_attrs2_: (
            list[OfflineNeuFoldedAttrsV2Part2] | list[dict[str, Any]] | None
        ),
    ) -> tuple[FrameArrayType, FrameArrayType, FrameArrayType]:
        """Generate three packages of half, full & folded neuron attributes."""
        dest_info = OfflineNeuDestInfoV2.model_validate(
            dest_info, strict=True
        ).model_dump()

        pkg_half_neu = np.array([], dtype=FRAME_DTYPE)
        pkg_full_neu = np.array([], dtype=FRAME_DTYPE)
        pkg_folded_neu = np.array([], dtype=FRAME_DTYPE)

        if full_attrs1 is not None:  # half
            full_attrs1 = OfflineNeuFullAttrsV2Part1.model_validate(
                full_attrs1, strict=True
            ).model_dump()
            pkg_half_neu = OfflineFrameGenV2._gen_pkg_half_neu(dest_info, full_attrs1)

            if full_attrs2 is not None:  # full
                full_attrs2 = OfflineNeuFullAttrsV2Part2.model_validate(
                    full_attrs2, strict=True
                ).model_dump()
                pkg_full_neu = OfflineFrameGenV2._gen_pkg_full_neu(
                    dest_info, full_attrs1, full_attrs2
                )

        if folded_attrs1 is not None and folded_attrs2_ is not None:  # folded
            folded_attrs1 = OfflineNeuFoldedAttrsV2Part1.model_validate(
                folded_attrs1, strict=True
            ).model_dump()
            folded_attrs2 = [
                OfflineNeuFoldedAttrsV2Part2.model_validate(
                    attrs2, strict=True
                ).model_dump()
                for attrs2 in folded_attrs2_
            ]
            pkg_folded_neu = OfflineFrameGenV2._gen_pkg_folded_neu(
                folded_attrs1, *folded_attrs2
            )
        else:
            raise ValueError("attributes of folded neuron are incomplete")

        return pkg_half_neu, pkg_full_neu, pkg_folded_neu

    @staticmethod
    def _gen_pkg_half_neu(
        dest_info: dict[str, Any], half_attrs: dict[str, Any]
    ) -> FrameArrayType:
        F = Off_Cfg3_V2.Full
        weight_skew_h11, weight_skew_l5 = bin_split(half_attrs["weight_skew"], 5, 11)
        # RAM[0][63:0]
        w1 = (
            (
                (weight_skew_l5 & F.Word1.WEIGHT_SKEW_LOW5_MASK)
                << F.Word1.WEIGHT_SKEW_LOW5_OFFSET
            )
            | (
                (half_attrs["weight_address_start"] & F.Word1.WEIGHT_ADDRESS_START_MASK)
                << F.Word1.WEIGHT_ADDRESS_START_OFFSET
            )
            | (
                (half_attrs["weight_address_end"] & F.Word1.WEIGHT_ADDRESS_END_MASK)
                << F.Word1.WEIGHT_ADDRESS_END_OFFSET
            )
            | (
                (half_attrs["output_type"] & F.Word1.OUTPUT_TYPE_MASK)
                << F.Word1.OUTPUT_TYPE_OFFSET
            )
            | (
                (half_attrs["fold_type"] & F.Word1.FOLD_TYPE_MASK)
                << F.Word1.FOLD_TYPE_OFFSET
            )
            | (
                (half_attrs["neuron_type"] & F.Word1.NEURON_TYPE_MASK)
                << F.Word1.NEURON_TYPE_OFFSET
            )
            | ((half_attrs["vjt"] & F.Word1.VJT_MASK) << F.Word1.VJT_OFFSET)
        )
        # RAM[0][127:64]
        w2 = (
            (
                (dest_info["tick_relative"] & F.Word2.TICK_RELATIVE_MASK)
                << F.Word2.TICK_RELATIVE_OFFSET
            )
            | (
                (dest_info["addr_axon"] & F.Word2.ADDR_AXON_MASK)
                << F.Word2.ADDR_AXON_OFFSET
            )
            | (
                (dest_info["addr_core_xy"] & F.Word2.ADDR_CORE_XY_MASK)
                << F.Word2.ADDR_CORE_XY_OFFSET
            )
            | (
                (dest_info["addr_core_x"] & F.Word2.ADDR_CORE_X_MASK)
                << F.Word2.ADDR_CORE_X_OFFSET
            )
            | (
                (dest_info["addr_core_y"] & F.Word2.ADDR_CORE_Y_MASK)
                << F.Word2.ADDR_CORE_Y_OFFSET
            )
            | (
                (dest_info["addr_copy_xy"] & F.Word2.ADDR_COPY_XY_MASK)
                << F.Word2.ADDR_COPY_XY_OFFSET
            )
            | (
                (dest_info["addr_copy_x"] & F.Word2.ADDR_COPY_X_MASK)
                << F.Word2.ADDR_COPY_X_OFFSET
            )
            | (
                (dest_info["addr_copy_y"] & F.Word2.ADDR_COPY_Y_MASK)
                << F.Word2.ADDR_COPY_Y_OFFSET
            )
            | (
                (weight_skew_h11 & F.Word2.WEIGHT_SKEW_HIGH11_MASK)
                << F.Word2.WEIGHT_SKEW_HIGH11_OFFSET
            )
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
        # RAM[1][63:0]
        w3 = (
            (
                (thres_pos_l20 & F.Word3.THRESHOLD_POS_LOW20_MASK)
                << F.Word3.THRESHOLD_POS_LOW20_OFFSET
            )
            | (
                (full_attrs2["lateral_inhibition"] & F.Word3.LATERAL_INHIBITION_MASK)
                << F.Word3.LATERAL_INHIBITION_OFFSET
            )
            | (
                (full_attrs2["leak_multi_sequence"] & F.Word3.LEAK_MULTI_SEQUENCE_MASK)
                << F.Word3.LEAK_MULTI_SEQUENCE_OFFSET
            )
            | (
                (full_attrs2["leak_multi_input"] & F.Word3.LEAK_MULTI_INPUT_MASK)
                << F.Word3.LEAK_MULTI_INPUT_OFFSET
            )
            | (
                (full_attrs2["leak_multi_mode"] & F.Word3.LEAK_MULTI_MODE_MASK)
                << F.Word3.LEAK_MULTI_MODE_OFFSET
            )
            | (
                (full_attrs2["leak_add_mode"] & F.Word3.LEAK_ADD_MODE_MASK)
                << F.Word3.LEAK_ADD_MODE_OFFSET
            )
            | (
                (full_attrs2["leak_tau"] & F.Word3.LEAK_TAU_MASK)
                << F.Word3.LEAK_TAU_OFFSET
            )
            | ((full_attrs2["leak_v"] & F.Word3.LEAK_V_MASK) << F.Word3.LEAK_V_OFFSET)
            | (
                (full_attrs2["weight_compress"] & F.Word3.WEIGHT_COMPRESS_MASK)
                << F.Word3.WEIGHT_COMPRESS_OFFSET
            )
            | (
                (full_attrs2["vjt_initial"] & F.Word3.VJT_INITIAL_MASK)
                << F.Word3.VJT_INITIAL_OFFSET
            )
        )
        # RAM[1][127:64]
        w4 = (
            (
                (full_attrs2["reset_mode"] & F.Word4.RESET_MODE_MASK)
                << F.Word4.RESET_MODE_OFFSET
            )
            | (
                (full_attrs2["reset_v"] & F.Word4.RESET_V_MASK)
                << F.Word4.RESET_V_OFFSET
            )
            | (
                (full_attrs2["threshold_neg_mode"] & F.Word4.THRESHOLD_NEG_MODE_MASK)
                << F.Word4.THRESHOLD_NEG_MODE_OFFSET
            )
            | (
                (full_attrs2["threshold_pos_mode"] & F.Word4.THRESHOLD_POS_MODE_MASK)
                << F.Word4.THRESHOLD_POS_MODE_OFFSET
            )
            | (
                (full_attrs2["threshold_neg"] & F.Word4.THRESHOLD_NEG_MASK)
                << F.Word4.THRESHOLD_NEG_OFFSET
            )
            | (
                (thres_pos_h12 & F.Word4.THRESHOLD_POS_HIGH12_MASK)
                << F.Word4.THRESHOLD_POS_HIGH12_OFFSET
            )
        )
        return np.r_[pkg_half_neu, w3, w4].astype(FRAME_DTYPE)

    @staticmethod
    def _gen_pkg_folded_neu(
        folded_attrs1: dict[str, Any], *folded_attrs2: dict[str, Any]
    ) -> FrameArrayType:
        F = Off_Cfg3_V2.Fold
        fold_skew_y_h9, fold_skew_y_l2 = bin_split(folded_attrs1["fold_skew_y"], 2, 9)
        # RAM[0][63:0]
        w1 = (
            (
                (fold_skew_y_l2 & F.Word1.FOLD_SKEW_Y_LOW2_MASK)
                << F.Word1.FOLD_SKEW_Y_LOW2_OFFSET
            )
            | (
                (folded_attrs1["fold_axon_xy"] & F.Word1.FOLD_AXON_XY_MASK)
                << F.Word1.FOLD_AXON_XY_OFFSET
            )
            | (
                (folded_attrs1["fold_axon_x"] & F.Word1.FOLD_AXON_X_MASK)
                << F.Word1.FOLD_AXON_X_OFFSET
            )
            | (
                (folded_attrs1["fold_axon_y"] & F.Word1.FOLD_AXON_Y_MASK)
                << F.Word1.FOLD_AXON_Y_OFFSET
            )
            | (
                (folded_attrs1["fold_number"] & F.Word1.FOLD_NUMBER_MASK)
                << F.Word1.FOLD_NUMBER_OFFSET
            )
        )
        # RAM[1][127:64]
        w2 = (
            (
                (folded_attrs1["fold_range_xy"] & F.Word2.FOLD_RANGE_XY_MASK)
                << F.Word2.FOLD_RANGE_XY_OFFSET
            )
            | (
                (folded_attrs1["fold_range_x"] & F.Word2.FOLD_RANGE_X_MASK)
                << F.Word2.FOLD_RANGE_X_OFFSET
            )
            | (
                (folded_attrs1["fold_range_y"] & F.Word2.FOLD_RANGE_Y_MASK)
                << F.Word2.FOLD_RANGE_Y_OFFSET
            )
            | (
                (folded_attrs1["fold_skew_xy"] & F.Word2.FOLD_SKEW_XY_MASK)
                << F.Word2.FOLD_SKEW_XY_OFFSET
            )
            | (
                (folded_attrs1["fold_skew_x"] & F.Word2.FOLD_SKEW_X_MASK)
                << F.Word2.FOLD_SKEW_X_OFFSET
            )
            | (
                (fold_skew_y_h9 & F.Word2.FOLD_SKEW_Y_HIGH9_MASK)
                << F.Word2.FOLD_SKEW_Y_HIGH9_OFFSET
            )
        )

        assert len(folded_attrs2) > 0
        v0 = np.array([item["fold_vjt_0"] for item in folded_attrs2], dtype=FRAME_DTYPE)
        v1 = np.array([item["fold_vjt_1"] for item in folded_attrs2], dtype=FRAME_DTYPE)
        v2 = np.array([item["fold_vjt_2"] for item in folded_attrs2], dtype=FRAME_DTYPE)
        v3 = np.array([item["fold_vjt_3"] for item in folded_attrs2], dtype=FRAME_DTYPE)

        # RAM[0][63:0]
        w3 = ((v1 & F.Word3.FOLD_VJT_1_MASK) << F.Word3.FOLD_VJT_1_OFFSET) | (
            (v0 & F.Word3.FOLD_VJT_0_MASK) << F.Word3.FOLD_VJT_0_OFFSET
        )
        # RAM[1][127:64]
        w4 = ((v3 & F.Word4.FOLD_VJT_3_MASK) << F.Word4.FOLD_VJT_3_OFFSET) | (
            (v2 & F.Word4.FOLD_VJT_2_MASK) << F.Word4.FOLD_VJT_2_OFFSET
        )
        v = np.zeros(len(w3) * 2, dtype=FRAME_DTYPE)
        v[0::2] = w3
        v[1::2] = w4

        return np.r_[w1, w2, v].astype(FRAME_DTYPE)

    @staticmethod
    def gen_config_frame3_weight_pkg(
        weight: np.ndarray,
        weight_width: DataWidth | Literal[1, 2, 4, 8],
        csc_compress: bool | CSCAccelerateMode = False,
    ) -> FrameArrayType:
        """Generate weight package for config frame type III."""
        assert weight.ndim == 1

        is_compress = csc_compress != CSCAccelerateMode.DISABLE
        if isinstance(weight_width, DataWidth):
            weight_width = 1 << weight_width.value
        else:
            weight_width = weight_width

        if is_compress:
            return weight_csc_pack(weight, weight_width)
        else:
            return weight_dense_pack(weight, weight_width)

    @staticmethod
    def gen_config_frame4(
        pkt_offset: CoordZXYOffset,
        input_array: FrameArrayType,
        pkt_ncopy: AERPacketZXYCopy = AERPacketZXYCopy(),
        start_addr: int = 0,
    ):
        # TODO
        pass

    @staticmethod
    def gen_work_frame1(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        timesteps: ArrayLike,
        axons: ArrayLike,
        target_lcn: LCN_EX,
        data: ArrayLike,
    ) -> FrameArrayType:
        F = Off_Work1_V2
        _data = np.asarray(data, dtype=PAYLOAD_DATA_DTYPE)
        ts = np.asarray(timesteps, dtype=FRAME_DTYPE).ravel()
        ax = np.asarray(axons, dtype=FRAME_DTYPE).ravel()

        if ax.size != ts.size:
            raise ValueError(
                f"the size of axons & timeslots are not equal, {ax.size} != {ts.size}."
            )
        if _data.size != ts.size:
            raise ValueError(
                f"the size of data & timeslots are not equal, {_data.size} != {ts.size}."
            )

        TS_WIDTH, AX_WIDTH = OfflineFrameGenV2.LCN_TO_TS_AXON_WIDTHS[target_lcn.value]
        ts_msb = (ts >> (TS_WIDTH - 1)) & F.TIMESTEP_HIGH7_MASK
        ts_low = ts & _mask(TS_WIDTH - 1)

        ts_ax_addr = (
            (ts_msb << F.TIMESTEP_HIGH7_OFFSET)
            | (ts_low << (F.AXON_ADDR_OFFSET + AX_WIDTH))
            | ((ax & _mask(AX_WIDTH)) << F.AXON_ADDR_OFFSET)
        )

        mask = np.flatnonzero(_data)
        data_u8 = _data.astype(PAYLOAD_DATA_DTYPE)
        frame_dest = get_frame_destV2(FH.WORK_TYPE1, pkt_offset, pkt_ncopy)
        return (frame_dest + ts_ax_addr[mask] + data_u8[mask]).astype(FRAME_DTYPE)

    @staticmethod
    def gen_control_frame1(
        pkt_offset: CoordZXYOffset, pkt_ncopy: AERPacketZXYCopy, n_timestep: int
    ) -> FrameArrayType:
        F = Off_Ctrl1_V2
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
        F = Off_Ctrl3_V2
        if thread_id > F.THREAD_ID_MASK:
            raise ValueError(f"'thread_id' out of range {F.THREAD_ID_MASK})")

        return FrameV2(FH.CTRL_TYPE3, pkt_offset, pkt_ncopy, thread_id).value


# class OfflineFrameDecoderV2:
#     @staticmethod
#     def decode_voltage(frame: FrameArrayType, target_lcn: LCN_EX) -> FrameArrayType:
#         pass


def weight_dense_pack(
    weight: np.ndarray, weight_width: Literal[1, 2, 4, 8]
) -> FrameArrayType:
    """Array uncompressed weights in RAM.

    weight[127:0] -> RAM[0][ 63: 0]
                     RAM[0][127:64]
    128/64/32/16 1/2/4/8-bit weights are stored in one address of RAM, little endian.
    """
    weight = weight.astype(np.uint8).ravel()
    align_size = 128 // weight_width
    aligned_size = math.ceil(weight.size / align_size) * align_size
    if (pad := aligned_size - weight.size) > 0:
        weight = np.pad(weight, (0, pad), constant_values=0)

    if weight_width == 8:
        return weight.view(FRAME_DTYPE)

    weights_per_byte = 8 // weight_width
    mask = _mask(weight_width)
    weight = weight.reshape(-1, weights_per_byte)  # (N, 8/ww)

    shifts = weight_width * np.arange(weights_per_byte, dtype=np.uint8)
    # Reduce to u8 (N,) and then view 8*u8 as u64
    reduced = np.bitwise_or.reduce((weight & mask) << shifts, axis=1, dtype=np.uint8)
    return reduced.view(FRAME_DTYPE)


def weight_csc_pack(
    weight: np.ndarray, weight_width: Literal[1, 2, 4, 8]
) -> FrameArrayType:
    """Arrange compressed weights according to CSC format.

    weight[127:0]: col_indices  +   weight
    1-bit           [127:16]        [ 6:0] stored 7 non-zero 1-bit weights
    2-bit           [127:16]        [13:0] stored 7 non-zero 2-bit weights
    4-bit           [127:32]        [23:0] stored 6 non-zero 4-bit weights
    8-bit           [127:32]        [39:0] stored 5 non-zero 8-bit weights

    NOTE: Non-zero values must be aligned in groups of 7/7/6/5. If all the values of a sparse weight    \
        are non-zero but need alignment, an exception will raise. If there is any 0, the first index    \
        pointing to 0 will be used as padding.
    """
    weight = weight.astype(np.uint8).ravel()
    # #N of non-zero weight stored in a single address of RAM
    N_NONZERO_WEIGHT_PER_ADDR = {1: 7, 2: 7, 4: 6, 8: 5}
    INDICES_ADDR_OFFSET = {1: 16, 2: 16, 4: 32, 8: 48}

    row_indices = np.flatnonzero(weight)
    w_nonzero = weight[row_indices]

    n_nonzero_w_per_addr = N_NONZERO_WEIGHT_PER_ADDR[weight_width]
    n_chunk = math.ceil(row_indices.size / n_nonzero_w_per_addr)

    if (pad := n_chunk * n_nonzero_w_per_addr - row_indices.size) > 0:
        if w_nonzero.size == weight.size:
            # NOTE: If the weight is all non-zero but needed to be padded, it's impossible to align.
            raise ValueError(
                f"the sparse weight cannot be aligned in groups of {n_nonzero_w_per_addr} "
                + "because there are all non-zero values."
            )

        # Get the first index of 0, pad with zeros & set the index pointing to it.
        idx_first_zero = np.where(weight == 0)[0][0]
        w_nonzero = np.pad(w_nonzero, (0, pad), constant_values=0)
        row_indices = np.pad(row_indices, (0, pad), constant_values=idx_first_zero)

    # (chunk, N)
    w_chunks = w_nonzero.reshape(n_chunk, n_nonzero_w_per_addr)
    # Index is no more than u16
    idx_chunks = row_indices.reshape(n_chunk, n_nonzero_w_per_addr).astype(np.uint16)

    w_shifts = weight_width * np.arange(n_nonzero_w_per_addr, dtype=np.uint8)
    idx_shifts = INDICES_ADDR_OFFSET[weight_width] + 16 * np.arange(
        n_nonzero_w_per_addr, dtype=np.uint8
    )
    w_mask = FRAME_DTYPE(_mask(weight_width))
    idx_mask = FRAME_DTYPE(_mask(16))

    # Reduce the weight & indices to (chunk,) u64
    w_reduced_chunk = np.bitwise_or.reduce(
        (w_chunks & w_mask) << w_shifts, axis=1, dtype=FRAME_DTYPE
    )

    n_idx_at_high = 4  # 4 indices will placed at high [127:64]
    idx_chunks_h = idx_chunks[:, -n_idx_at_high:]
    idx_chunks_l = idx_chunks[:, :-n_idx_at_high]
    idx_shifs_h = idx_shifts[-n_idx_at_high:] - 64
    idx_shifs_l = idx_shifts[:-n_idx_at_high]

    idx_reduced_chunk_h = np.bitwise_or.reduce(
        (idx_chunks_h & idx_mask) << idx_shifs_h, axis=1, dtype=FRAME_DTYPE
    )
    idx_reduced_chunk_l = np.bitwise_or.reduce(
        (idx_chunks_l & idx_mask) << idx_shifs_l, axis=1, dtype=FRAME_DTYPE
    )

    result = np.zeros((2 * n_chunk,), dtype=FRAME_DTYPE)
    # Little endian
    result[0::2] = idx_reduced_chunk_l + w_reduced_chunk
    result[1::2] = idx_reduced_chunk_h
    return result


def weight_dense_unpack(
    frames: FrameArrayType,
    weight_width: Literal[1, 2, 4, 8],
    signed: bool,
    original_size: int,
) -> NDArray[np.int8 | np.uint8]:
    w_per_u64 = 64 // weight_width
    mask = _mask(weight_width)

    shifts = weight_width * np.arange(w_per_u64, dtype=np.uint8)
    extracted = ((frames[:, np.newaxis] >> shifts) & mask).ravel()

    # Omit the part of padding
    result = extracted[:original_size]

    if signed and weight_width < 8:
        signbit = 1 << (weight_width - 1)
        result = result.astype(np.int8)
        result[result >= signbit] -= 1 << weight_width

    return result.astype(np.int8 if signed else np.uint8)


def weight_csc_unpack(
    frames: FrameArrayType,
    weight_width: Literal[1, 2, 4, 8],
    signed: bool,
    original_size: int,
) -> NDArray[np.int8 | np.uint8]:
    # #N of non-zero weight stored in a single address of RAM
    N_NONZERO_WEIGHT_PER_ADDR = {1: 7, 2: 7, 4: 6, 8: 5}
    INDICES_ADDR_OFFSET = {1: 16, 2: 16, 4: 32, 8: 48}
    n_nonzero_w_per_addr = N_NONZERO_WEIGHT_PER_ADDR[weight_width]

    assert frames.size % 2 == 0

    w_shifts = weight_width * np.arange(n_nonzero_w_per_addr, dtype=np.uint8)
    idx_shifts = INDICES_ADDR_OFFSET[weight_width] + 16 * np.arange(
        n_nonzero_w_per_addr, dtype=np.uint8
    )
    w_mask = _mask(weight_width)
    idx_mask = _mask(16)

    # (chunk,)
    chunk_low64 = frames[0::2]
    chunk_high64 = frames[1::2]

    weights = ((chunk_low64[:, np.newaxis] >> w_shifts[np.newaxis, :]) & w_mask).astype(
        np.uint8
    )

    n_idx_at_high = 4  # 4 indices will placed at high [127:64]
    idx_shifs_h = idx_shifts[-n_idx_at_high:] - 64
    idx_shifs_l = idx_shifts[:-n_idx_at_high]
    indices_h = (chunk_high64[:, np.newaxis] >> idx_shifs_h) & idx_mask
    indices_l = (chunk_low64[:, np.newaxis] >> idx_shifs_l) & idx_mask
    # Little endian
    indices = np.hstack([indices_l, indices_h])

    if signed and weight_width < 8:
        signbit = 1 << (weight_width - 1)
        weights = weights.astype(np.int8)
        weights[weights >= signbit] -= 1 << weight_width

    result = np.zeros(original_size, dtype=np.int8 if signed else np.uint8)
    result[indices] = weights
    return result
