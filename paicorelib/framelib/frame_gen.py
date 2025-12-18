from collections.abc import Sequence
from typing import Any, ClassVar, overload

import numpy as np
from numpy.typing import ArrayLike

from ..coordinate import ChipCoord, Coord
from ..coordinate import ReplicationId as RId
from ..ram_model import OfflineNeuAttrs as OffNeuAttrs
from ..ram_model import OfflineNeuDestInfo as OffNeuDestInfo
from ..ram_model import OnlineNeuAttrs as OnNeuAttrs
from ..ram_model import OnlineNeuDestInfo as OnNeuDestInfo
from ..reg_defs import LCN_EX, CoreType
from ..reg_defs import WeightWidth as WW
from ..reg_model import OfflineCoreReg as OffCoreReg
from ..reg_model import OnlineCoreReg as OnCoreReg
from ..routing_defs import _rid_unset
from .frames import *
from .types import *

__all__ = ["OfflineFrameGen", "OnlineFrameGen", "ChipFrameGen", "OfflineFrameGenV2"]

# -------------------------------------------------------------------------
# 补充完整的 V2.5 引用 (保持原有风格)
# -------------------------------------------------------------------------
from .frame_defs import (
    FrameFormatV2 as FFV2,
    FrameHeaderV2 as FHV2,
    # Config Frames
    OfflineConfigFrame1FormatV2 as Off_Cfg1_V2,
    OfflineConfigFrame2FormatV2 as Off_Cfg2_V2,
    OfflineConfigFrame3FormatV2 as Off_Cfg3_V2,
    OfflineConfigFrame4FormatV2 as Off_Cfg4_V2,
    # Work Frames
    OfflineWorkFrame1FormatV2 as Off_WF1F_V2,
    OfflineWorkFrame2FormatV2 as Off_WF2F_V2,
    # Control Frames
    OfflineControlFrame1Format as Off_CF1F,
    OfflineControlFrame2Format as Off_CF2F,
    OfflineControlFrame3Format as Off_CF3F,
)



class OfflineFrameGen:
    """Offline frame generator."""

    @staticmethod
    def gen_config_frame1(
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        /,
        random_seed: IntScalarType,
    ) -> OfflineConfigFrame1:
        return OfflineConfigFrame1(chip_coord, core_coord, rid, int(random_seed))

    @staticmethod
    def gen_config_frame2(
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        /,
        core_reg: OffCoreReg | dict[str, Any],
    ) -> OfflineConfigFrame2:
        return OfflineConfigFrame2(chip_coord, core_coord, rid, core_reg)

    @overload
    @staticmethod
    def gen_config_frame3(
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        /,
        neu_start_addr: int,
        n_neuron: int,
        attrs: OffNeuAttrs,
        dest_info: OffNeuDestInfo,
        repeat: int = 1,
    ) -> OfflineConfigFrame3: ...

    @overload
    @staticmethod
    def gen_config_frame3(
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        /,
        neu_start_addr: int,
        n_neuron: int,
        attrs: dict[str, Any],
        dest_info: dict[str, Any],
        repeat: int = 1,
    ) -> OfflineConfigFrame3: ...

    @staticmethod
    def gen_config_frame3(
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        /,
        neu_start_addr: int,
        n_neuron: int,
        attrs: OffNeuAttrs | dict[str, Any],
        dest_info: OffNeuDestInfo | dict[str, Any],
        repeat: int = 1,
    ) -> OfflineConfigFrame3:
        return OfflineConfigFrame3(
            chip_coord,
            core_coord,
            rid,
            neu_start_addr,
            n_neuron,
            attrs,
            dest_info,
            repeat,
        )

    @staticmethod
    def gen_config_frame4(
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        /,
        neu_start_addr: int,
        n_data_package: int,
        weight_ram: FrameArrayType,
    ) -> OfflineConfigFrame4:
        return OfflineConfigFrame4(
            chip_coord, core_coord, rid, neu_start_addr, n_data_package, weight_ram
        )

    @staticmethod
    def gen_testin_frame1(
        chip_coord: ChipCoord, core_coord: Coord, rid: RId, /
    ) -> OfflineTestInFrame1:
        return OfflineTestInFrame1(chip_coord, core_coord, rid)

    @staticmethod
    def gen_testout_frame1(
        test_chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        random_seed: IntScalarType,
    ) -> OfflineTestOutFrame1:
        return OfflineTestOutFrame1(test_chip_coord, core_coord, rid, int(random_seed))

    @staticmethod
    def gen_testin_frame2(
        chip_coord: ChipCoord, core_coord: Coord, rid: RId, /
    ) -> OfflineTestInFrame2:
        return OfflineTestInFrame2(chip_coord, core_coord, rid)

    @staticmethod
    def gen_testout_frame2(
        test_chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        core_reg: OffCoreReg | dict[str, Any],
    ) -> OfflineTestOutFrame2:
        return OfflineTestOutFrame2(test_chip_coord, core_coord, rid, core_reg)

    @staticmethod
    def gen_testin_frame3(
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        /,
        neu_start_addr: int,
        n_package: int,
    ) -> OfflineTestInFrame3:
        return OfflineTestInFrame3(
            chip_coord, core_coord, rid, neu_start_addr, n_package
        )

    @overload
    @staticmethod
    def gen_testout_frame3(
        test_chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        /,
        neu_start_addr: int,
        n_neuron: int,
        attrs: OffNeuAttrs | dict[str, Any],
        dest_info: OffNeuDestInfo | dict[str, Any],
        lcn_ex: LCN_EX,
        weight_width: WW,
    ) -> OfflineTestOutFrame3: ...

    @overload
    @staticmethod
    def gen_testout_frame3(
        test_chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        /,
        neu_start_addr: int,
        n_neuron: int,
        attrs: OffNeuAttrs | dict[str, Any],
        dest_info: OffNeuDestInfo | dict[str, Any],
        *,
        repeat: int,
    ) -> OfflineTestOutFrame3: ...

    @staticmethod
    def gen_testout_frame3(
        test_chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        /,
        neu_start_addr: int,
        n_neuron: int,
        attrs: OffNeuAttrs | dict[str, Any],
        dest_info: OffNeuDestInfo | dict[str, Any],
        lcn_ex: LCN_EX | None = None,
        weight_width: WW | None = None,
        *,
        repeat: int | None = None,
    ) -> OfflineTestOutFrame3:
        if lcn_ex is not None and weight_width is not None:
            repeat = 1 << (lcn_ex + weight_width)
        else:
            assert repeat is not None

        return OfflineTestOutFrame3(
            test_chip_coord,
            core_coord,
            rid,
            neu_start_addr,
            n_neuron,
            attrs,
            dest_info,
            repeat,
        )

    @staticmethod
    def gen_testin_frame4(
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        /,
        neu_start_addr: int,
        n_package: int,
    ) -> OfflineTestInFrame4:
        return OfflineTestInFrame4(
            chip_coord, core_coord, rid, neu_start_addr, n_package
        )

    @staticmethod
    def gen_testout_frame4(
        test_chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        /,
        neu_start_addr: int,
        n_data_package: int,
        weight_ram: FrameArrayType,
    ) -> OfflineTestOutFrame4:
        return OfflineTestOutFrame4(
            test_chip_coord, core_coord, rid, neu_start_addr, n_data_package, weight_ram
        )

    @staticmethod
    def gen_work_frame1(
        one_input_node: dict[str, Any], data: ArrayLike
    ) -> FrameArrayType:
        """Generate the common part of the input spike frames by given the info of one input node.

        Args:
            one_input_node: a dictionary of a single input node that points to offline cores.
            data: the input data.
        """
        common_frame_dest = OfflineWorkFrame1._frame_dest_reorganized(one_input_node)
        _data = np.asarray(data, dtype=PAYLOAD_DATA_DTYPE)

        return OfflineFrameGen.gen_work_frame1_fast(common_frame_dest, _data)

    @staticmethod
    def gen_work_frame1_fast(
        frame_dest_info: FrameArrayType, data: PayloadDataType
    ) -> FrameArrayType:
        if frame_dest_info.size != data.size:
            raise ValueError(
                f"the size of frame dest info & data are not equal, {frame_dest_info.size} != {data.size}."
            )

        mask = np.flatnonzero(data)
        return frame_dest_info[mask] + data[mask]

    @staticmethod
    def gen_work_frame2(chip_coord: ChipCoord, /, n_sync: int) -> OfflineWorkFrame2:
        return OfflineWorkFrame2(chip_coord, n_sync)

    @staticmethod
    def gen_work_frame3(chip_coord: ChipCoord) -> OfflineWorkFrame3:
        return OfflineWorkFrame3(chip_coord)

    @staticmethod
    def gen_work_frame4(chip_coord: ChipCoord) -> OfflineWorkFrame4:
        return OfflineWorkFrame4(chip_coord)


class OnlineFrameGen:
    """Online frame generator."""

    @staticmethod
    def gen_config_frame1(
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        /,
        lut: LUTDataType | None = None,
    ) -> OnlineConfigFrame1:
        return OnlineConfigFrame1(chip_coord, core_coord, rid, lut)

    @staticmethod
    def gen_config_frame2(
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        /,
        core_reg: OnCoreReg | dict[str, Any],
    ) -> OnlineConfigFrame2:
        return OnlineConfigFrame2(chip_coord, core_coord, rid, core_reg)

    @overload
    @staticmethod
    def gen_config_frame3(
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        /,
        neu_start_addr: int,
        n_neuron: int,
        attrs: OnNeuAttrs,
        dest_info: OnNeuDestInfo,
        weight_width: WW,
    ) -> OnlineConfigFrame3: ...

    @overload
    @staticmethod
    def gen_config_frame3(
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        /,
        neu_start_addr: int,
        n_neuron: int,
        attrs: dict[str, Any],
        dest_info: dict[str, Any],
        weight_width: WW,
    ) -> OnlineConfigFrame3: ...

    @staticmethod
    def gen_config_frame3(
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        /,
        neu_start_addr: int,
        n_neuron: int,
        attrs: OnNeuAttrs | dict[str, Any],
        dest_info: OnNeuDestInfo | dict[str, Any],
        weight_width: WW,
    ) -> OnlineConfigFrame3:
        return OnlineConfigFrame3(
            chip_coord,
            core_coord,
            rid,
            neu_start_addr,
            n_neuron,
            attrs,
            dest_info,
            weight_width,
        )

    @staticmethod
    def gen_config_frame4(
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        /,
        neu_start_addr: int,
        n_data_package: int,
        weight_ram: FrameArrayType,
    ) -> OnlineConfigFrame4:
        return OnlineConfigFrame4(
            chip_coord, core_coord, rid, neu_start_addr, n_data_package, weight_ram
        )

    @staticmethod
    def gen_testin_frame1(
        chip_coord: ChipCoord, core_coord: Coord, rid: RId
    ) -> OnlineTestInFrame1:
        return OnlineTestInFrame1(chip_coord, core_coord, rid)

    @staticmethod
    def gen_testout_frame1(
        test_chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        lut: LUTDataType | None = None,
    ) -> OnlineTestOutFrame1:
        return OnlineTestOutFrame1(test_chip_coord, core_coord, rid, lut)

    @staticmethod
    def gen_testin_frame2(
        chip_coord: ChipCoord, core_coord: Coord, rid: RId
    ) -> OnlineTestInFrame2:
        return OnlineTestInFrame2(chip_coord, core_coord, rid)

    @staticmethod
    def gen_testout_frame2(
        test_chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        core_reg: OnCoreReg | dict[str, Any],
    ) -> OnlineTestOutFrame2:
        return OnlineTestOutFrame2(test_chip_coord, core_coord, rid, core_reg)

    @staticmethod
    def gen_testin_frame3(
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        /,
        neu_start_addr: int,
        n_package: int,
    ) -> OnlineTestInFrame3:
        return OnlineTestInFrame3(
            chip_coord, core_coord, rid, neu_start_addr, n_package
        )

    @overload
    @staticmethod
    def gen_testout_frame3(
        test_chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        /,
        neu_start_addr: int,
        n_neuron: int,
        attrs: OnNeuAttrs,
        dest_info: OnNeuDestInfo,
        weight_width: WW,
    ) -> OnlineTestOutFrame3: ...

    @overload
    @staticmethod
    def gen_testout_frame3(
        test_chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        /,
        neu_start_addr: int,
        n_neuron: int,
        attrs: dict[str, Any],
        dest_info: dict[str, Any],
        weight_width: WW,
    ) -> OnlineTestOutFrame3: ...

    @staticmethod
    def gen_testout_frame3(
        test_chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        /,
        neu_start_addr: int,
        n_neuron: int,
        attrs: OnNeuAttrs | dict[str, Any],
        dest_info: OnNeuDestInfo | dict[str, Any],
        weight_width: WW,
    ) -> OnlineTestOutFrame3:
        return OnlineTestOutFrame3(
            test_chip_coord,
            core_coord,
            rid,
            neu_start_addr,
            n_neuron,
            attrs,
            dest_info,
            weight_width,
        )

    @staticmethod
    def gen_testin_frame4(
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        /,
        neu_start_addr: int,
        n_package: int,
    ) -> OnlineTestInFrame4:
        return OnlineTestInFrame4(
            chip_coord, core_coord, rid, neu_start_addr, n_package
        )

    @staticmethod
    def gen_testout_frame4(
        test_chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        /,
        neu_start_addr: int,
        n_data_package: int,
        weight_ram: FrameArrayType,
    ) -> OnlineTestOutFrame4:
        return OnlineTestOutFrame4(
            test_chip_coord, core_coord, rid, neu_start_addr, n_data_package, weight_ram
        )

    @staticmethod
    def gen_work_frame1_1(
        one_input_node: dict[str, Any], data: ArrayLike
    ) -> FrameArrayType:
        """Generate the common part of the input spike frames by given the info of one input node.

        Args:
            one_input_node: a dictionary of a single input node that points to online cores.
        """
        common_frame_dest = OnlineWorkFrame1_1._frame_dest_reorganized(one_input_node)
        _data = np.asarray(data, dtype=PAYLOAD_DATA_DTYPE)

        return OnlineFrameGen.gen_work_frame1_1_fast(common_frame_dest, _data)

    @staticmethod
    def gen_work_frame1_1_fast(
        frame_dest_info: FrameArrayType, data: PayloadDataType
    ) -> FrameArrayType:
        if frame_dest_info.size != data.size:
            raise ValueError(
                f"the size of frame dest info & mask are not equal, {frame_dest_info.size} != {data.size}."
            )

        mask = np.flatnonzero(data)
        # Only where there is 1 returns.
        return frame_dest_info[mask]

    @staticmethod
    def gen_work_frame1_2(
        chip_coord: ChipCoord, coord: Coord, rid: RId = _rid_unset(), /
    ) -> OnlineWorkFrame1_2:
        return OnlineWorkFrame1_2(chip_coord, coord, rid)

    @staticmethod
    def gen_work_frame1_3(
        chip_coord: ChipCoord, coord: Coord, rid: RId = _rid_unset(), /
    ) -> OnlineWorkFrame1_3:
        return OnlineWorkFrame1_3(chip_coord, coord, rid)

    @staticmethod
    def gen_work_frame1_4(
        chip_coord: ChipCoord, coord: Coord, rid: RId = _rid_unset(), /
    ) -> OnlineWorkFrame1_4:
        return OnlineWorkFrame1_4(chip_coord, coord, rid)

    @staticmethod
    def gen_work_frame2(chip_coord: ChipCoord, /, n_sync: int) -> OnlineWorkFrame2:
        return OnlineWorkFrame2(chip_coord, n_sync)

    @staticmethod
    def gen_work_frame3(chip_coord: ChipCoord) -> OnlineWorkFrame3:
        return OnlineWorkFrame3(chip_coord)

    @staticmethod
    def gen_work_frame4(chip_coord: ChipCoord) -> OnlineWorkFrame4:
        return OnlineWorkFrame4(chip_coord)


class ChipFrameGen:
    fgen_handler: ClassVar[dict[CoreType, OfflineFrameGen | OnlineFrameGen]] = {
        CoreType.OFFLINE: OfflineFrameGen(),
        CoreType.ONLINE: OnlineFrameGen(),
    }

    @classmethod
    def gen_magic_init_frame(
        cls,
        chip_coord: ChipCoord,
        core_coord: Coord | Sequence[Coord],
        core_rid: RId | Sequence[RId] = _rid_unset(),
        redundant_init: bool = True,
    ) -> tuple[FrameArrayType, FrameArrayType]:
        """Magic initialization frames for PAICORE. DO NOT MODIFY!

        Args:
            chip_coord: coordinate of the target chip.
            core_coord: coordinates of the target cores.
            redundant_init: whether to use redundant initialization frames, in case of failure.

        If use redundant initialization frames, the magic frames are composed of:
            1. [config1[0] of core #1] + [init frame] + [config1[0] of core #2] + [init frame] + ...\
                + [config1[0] of core #N] + [init frame]

            2. [config1[1] of core #1] + [config1[2] of core #1] + [config1[1] of core #2] + ... +  \
                [config1[2] of core #2] + [config1[1] of core #N] + [config1[2] of core #N]
            3. [work1[0] of core #1] + [work1[0] of core #2] + ... + [work1[0] of core #N]

        Else,
            1. [config1[0] of core #1] + [config1[0] of core #2] + ... + [config1[0] of core #N] +  \
                [init frame]

            2, 3 remain the same.

        Returns: two parts of magic frames.
        """
        if isinstance(core_coord, Coord):
            _core_coord = (core_coord,)
        else:
            _core_coord = core_coord

        if isinstance(core_rid, RId):
            _core_rid = (core_rid,) * len(_core_coord)
        else:
            _core_rid = core_rid

        magic_frame_cf_1 = []
        magic_frame_cf_2 = []
        magic_frame_wf = []
        init_frame = OfflineWorkFrame4(chip_coord)

        for coord, rid in zip(_core_coord, _core_rid, strict=True):
            # NOTE: The length of config type I for online & offline cores are not the same.
            if coord.core_type == CoreType.OFFLINE:
                config1 = OfflineConfigFrame1(chip_coord, coord, rid, 0)
            else:
                config1 = OnlineConfigFrame1(chip_coord, coord, rid, None)

            work1 = OfflineWorkFrame1(chip_coord, coord, rid, 0, 0, 0)

            magic_frame_cf_1.append(config1.value[0])
            if redundant_init:
                magic_frame_cf_1.append(init_frame.value[0])

            magic_frame_cf_2.extend(config1.value[1:])
            magic_frame_wf.append(work1.value[0])

        if not redundant_init:
            magic_frame_cf_1.append(init_frame.value[0])

        magic_frame_cf_2.extend(magic_frame_wf)

        return np.asarray(magic_frame_cf_1, dtype=FRAME_DTYPE), np.asarray(
            magic_frame_cf_2, dtype=FRAME_DTYPE
        )
class OfflineFrameGenV2:
    """Offline frame generator for FFV2 format (V2.5 Protocol).
    
    Strictly follows V2.5 definition:
    - Packet Protocol for Config/Test (Header + Body).
    - Single Frame for Work/Control.
    """

    # =========================================================================
    # 1. 基础工具 (Packer)
    # =========================================================================

    @staticmethod
    def _pack_frame(
        header: int,
        core_addr: int,
        copy_addr: int,
        payload: int,
    ) -> FrameArrayType:
        """按 FFV2 布局打包单个标准帧 (Header + Addrs + 24bit Payload)。
        注意：使用 FFV2 (FrameFormatV2) 定义的常量。
        """
        frame = (
            ((header & FFV2.GENERAL_HEADER_MASK) << FFV2.GENERAL_HEADER_OFFSET)
            | ((core_addr & FFV2.GENERAL_CORE_ADDR_MASK) << FFV2.GENERAL_CORE_ADDR_OFFSET)
            | ((copy_addr & FFV2.GENERAL_COPY_ADDR_MASK) << FFV2.GENERAL_COPY_ADDR_OFFSET)
            | (payload & FFV2.GENERAL_PAYLOAD_MASK)
        )
        return np.asarray([FRAME_DTYPE(frame)], dtype=FRAME_DTYPE)

    @staticmethod
    def gen_packet(
        core_addr: int,
        copy_addr: int,
        frame_type: int,
        start_addr: int,
        payload_data: list[int],
        is_test_request: bool = False
    ) -> FrameArrayType:
        """
        生成符合 V2.5 数据包协议的帧序列。
        
        Args:
            frame_type: FrameHeaderV2 枚举值
            start_addr: 起始地址
            payload_data: 64-bit 数据列表 (Body)
            is_test_request: 是否为测试读请求 (True: Type=1, False: Type=0)
        """
        frames = []
        n_package = len(payload_data)
        
        if start_addr > 0x1FF:
            raise ValueError(f"start_addr {hex(start_addr)} exceeds 9 bits")
        if n_package > 0x3FFF:
            raise ValueError(f"Number of packages {n_package} exceeds 14 bits")
            
        # V2.5 Protocol: [23] Package Type (0: Config/TestOut, 1: TestIn)
        pkg_type_bit = 1 if is_test_request else 0
        
        header_payload = (
            (pkg_type_bit << FFV2.GENERAL_PACKAGE_TYPE_OFFSET) | 
            ((start_addr & FFV2.GENERAL_PACKAGE_NEU_START_ADDR_MASK) << FFV2.GENERAL_PACKAGE_NEU_START_ADDR_OFFSET) |
            ((n_package & FFV2.GENERAL_PACKAGE_NUM_MASK) << FFV2.GENERAL_PACKAGE_NUM_OFFSET)
        )
        
        # Frame 0: Packet Header
        frames.append(OfflineFrameGenV2._pack_frame(
            frame_type, core_addr, copy_addr, header_payload
        ))
        
        # Frame 1..N: Data Body (仅当非请求帧时生成)
        if not is_test_request:
            for i, data in enumerate(payload_data):
                if data < 0 or data > 0xFFFFFFFFFFFFFFFF:
                     raise ValueError(f"Data at index {i} out of 64-bit range: {hex(data)}")
                frames.append(np.asarray([data], dtype=FRAME_DTYPE))
            
        return np.concatenate(frames)

    # =========================================================================
    # 2. 配置帧生成 (Configuration Frames)
    # =========================================================================

    @staticmethod
    def gen_core_config(
        core_addr: int,
        copy_addr: int,
        params: dict[str, int]
    ) -> FrameArrayType:
        """生成 Type 1 (Core Parameters) 配置包。"""
        F = Off_Cfg1_V2
        
        # Word 0
        w0 = (
            ((params.get('snn_ann', 0) & 1) << F.Word0.SNN_ANN_OFFSET) |
            ((params.get('max_pooling', 0) & 1) << F.Word0.MAX_POOLING_OFFSET) |
            ((params.get('add_potential', 0) & 1) << F.Word0.ADD_POTENTIAL_OFFSET) |
            ((params.get('zero_output', 0) & 1) << F.Word0.ZERO_OUTPUT_OFFSET) |
            ((params.get('input_sign', 0) & 1) << F.Word0.INPUT_SIGH_OFFSET) |
            ((params.get('input_width', 0) & 0x3) << F.Word0.INPUT_WIDTH_OFFSET) |
            ((params.get('output_sign', 0) & 1) << F.Word0.OUTPUT_SIGH_OFFSET) |
            ((params.get('output_width', 0) & 0x3) << F.Word0.OUTPUT_WIDTH_OFFSET) |
            ((params.get('weight_sign', 0) & 1) << F.Word0.WEIGHT_SIGH_OFFSET) |
            ((params.get('weight_width', 0) & 0x3) << F.Word0.WEIGHT_WIDTH_OFFSET) |
            ((params.get('lcn', 0) & 0xF) << F.Word0.LCN_OFFSET) |
            ((params.get('target_lcn', 0) & 0xF) << F.Word0.TARGET_LCN_OFFSET) |
            ((params.get('axon_skew', 0) & 0xFFFF) << F.Word0.AXON_SKEW_OFFSET) |
            ((params.get('neuron_number', 0) & 0x1FFF) << F.Word0.NEURON_NUMBER_OFFSET) |
            ((params.get('test_core_xy', 0) & 0x3F) << F.Word0.TEST_CORE_XY_OFFSET) |
            ((params.get('test_core_x', 0) & 0x3F) << F.Word0.TEST_CORE_X_OFFSET) |
            ((params.get('test_core_y_high', 0) & 0x3) << F.Word0.TEST_CORE_Y_HIGH2_OFFSET)
        )

        # Word 1
        w1 = (
            ((params.get('test_core_y_low', 0) & 0xF) << F.Word1.TEST_CORE_Y_LOW4_OFFSET) |
            ((params.get('global_send', 0) & 0x7F) << F.Word1.GLOBAL_SEND_OFFSET) |
            ((params.get('csc_accelerate', 0) & 1) << F.Word1.CSC_ACCELERATE_OFFSET) |
            ((params.get('global_receive', 0) & 0x3F) << F.Word1.GLOBAL_RECEIVE_OFFSET) |
            ((params.get('thread_number', 0) & 0x3FF) << F.Word1.THREAD_NUMBER_OFFSET) |
            ((params.get('busy_cycle', 0) & 0xFFF) << F.Word1.BUSY_CYCLE_OFFSET) |
            ((params.get('delay_cycle', 0) & 0xFFFF) << F.Word1.DELAY_CYCLE_OFFSET) |
            ((params.get('width_cycle', 0) & 0xFF) << F.Word1.WIDTH_CYCLE_OFFSET)
        )

        # Word 2
        w2 = (
            ((params.get('tick_start', 0) & 0xFFFF) << F.Word2.TICK_START_OFFSET) |
            ((params.get('tick_duration', 0) & 0xFFFFFFFF) << F.Word2.TICK_DURATION_OFFSET) |
            ((params.get('tick_initial', 0) & 0xFFFF) << F.Word2.TICK_INITIAL_OFFSET)
        )

        return OfflineFrameGenV2.gen_packet(
            core_addr, copy_addr, int(FHV2.CONFIG_TYPE1), 0, [w0, w1, w2]
        )

    @staticmethod
    def gen_lut_config(
        core_addr: int,
        copy_addr: int,
        lut_entries: list[dict[str, int]],
        start_index: int = 0
    ) -> FrameArrayType:
        """生成 Type 2 (LUT SRAM) 配置包。"""
        F = Off_Cfg2_V2
        payloads = []
        
        for entry in lut_entries:
            w = (
                ((entry.get('potential', 0) & 0xFFFFFFFF) << F.POTENTIAL_OFFSET) |
                ((entry.get('activation', 0) & 0xFF) << F.ACTIVATION_OFFSET)
            )
            payloads.append(w)
            
        return OfflineFrameGenV2.gen_packet(
            core_addr, copy_addr, int(FHV2.CONFIG_TYPE2), start_index, payloads
        )

    @staticmethod
    def gen_neuron_config(
        core_addr: int,
        copy_addr: int,
        neurons: list[dict[str, int]],
        start_addr: int = 0,
        mode: str = 'full'
    ) -> FrameArrayType:
        """生成 Type 3 (Neuron SRAM) 配置包。"""
        payloads = []
        
        if mode == 'full':
            F = Off_Cfg3_V2.Full
            for neu in neurons:
                # Word 0
                w0 = (
                    ((neu.get('tick_relative', 0) & 0xFF) << F.Word0.TICK_RELATIVE_OFFSET) |
                    ((neu.get('addr_axon', 0) & 0x1FF) << F.Word0.ADDR_AXON_OFFSET) |
                    ((neu.get('addr_core_xy', 0) & 0x3F) << F.Word0.ADDR_CORE_XY_OFFSET) |
                    ((neu.get('addr_core_x', 0) & 0x3F) << F.Word0.ADDR_CORE_X_OFFSET) |
                    ((neu.get('addr_core_y', 0) & 0x3F) << F.Word0.ADDR_CORE_Y_OFFSET) |
                    ((neu.get('addr_copy_xy', 0) & 0x3F) << F.Word0.ADDR_COPY_XY_OFFSET) |
                    ((neu.get('addr_copy_x', 0) & 0x3F) << F.Word0.ADDR_COPY_X_OFFSET) |
                    ((neu.get('addr_copy_y', 0) & 0x3F) << F.Word0.ADDR_COPY_Y_OFFSET) |
                    ((neu.get('weight_skew_high', 0) & 0x7FF) << F.Word0.WEIGHT_SKEW_HIGH_OFFSET)
                )
                # Word 1
                w1 = (
                    ((neu.get('weight_skew_low', 0) & 0x1F) << F.Word1.WEIGHT_SKEW_LOW_OFFSET) |
                    ((neu.get('weight_addr_start', 0) & 0xFFF) << F.Word1.WEIGHT_ADDRESS_START_OFFSET) |
                    ((neu.get('weight_addr_end', 0) & 0xFFF) << F.Word1.WEIGHT_ADDRESS_END_OFFSET) |
                    ((neu.get('output_type', 0) & 1) << F.Word1.OUTPUT_TYPE_OFFSET) |
                    ((neu.get('fold_type', 0) & 1) << F.Word1.FOLD_TYPE_OFFSET) |
                    ((neu.get('neuron_type', 0) & 1) << F.Word1.NEURON_TYPE_OFFSET) |
                    ((neu.get('vjt', 0) & 0xFFFFFFFF) << F.Word1.VJT_OFFSET)
                )
                # Word 2
                w2 = (
                    ((neu.get('reset_mode', 0) & 0x3) << F.Word2.RESET_MODE_OFFSET) |
                    ((neu.get('reset_v', 0) & 0xFFFF) << F.Word2.RESET_V_OFFSET) |
                    ((neu.get('thres_neg_mode', 0) & 1) << F.Word2.THRESHOLD_NEG_MODE_OFFSET) |
                    ((neu.get('thres_pos_mode', 0) & 1) << F.Word2.THRESHOLD_POS_MODE_OFFSET) |
                    ((neu.get('thres_neg', 0) & 0xFFFFFFFF) << F.Word2.THRESHOLD_NEG_OFFSET) |
                    ((neu.get('thres_pos_hi', 0) & 0xFFF) << F.Word2.THRESHOLD_POS_HI_OFFSET)
                )
                # Word 3
                w3 = (
                    ((neu.get('thres_pos_lo', 0) & 0xFFFFF) << F.Word3.THRESHOLD_POS_LO_OFFSET) |
                    ((neu.get('lateral_inhibit', 0) & 1) << F.Word3.LATERAL_INHIBITION_OFFSET) |
                    ((neu.get('leak_multi_seq', 0) & 1) << F.Word3.LEAK_MULTI_SEQUENCE_OFFSET) |
                    ((neu.get('leak_multi_in', 0) & 1) << F.Word3.LEAK_MULTI_INPUT_OFFSET) |
                    ((neu.get('leak_multi_mode', 0) & 1) << F.Word3.LEAK_MULTI_MODE_OFFSET) |
                    ((neu.get('leak_add_mode', 0) & 1) << F.Word3.LEAK_ADD_MODE_OFFSET) |
                    ((neu.get('leak_tau', 0) & 0x3F) << F.Word3.LEAK_TAU_OFFSET) |
                    ((neu.get('leak_v', 0) & 0xFFFFF) << F.Word3.LEAK_V_OFFSET) |
                    ((neu.get('weight_compress', 0) & 1) << F.Word3.WEIGHT_COMPRESS_OFFSET) |
                    ((neu.get('vjt_initial', 0) & 0xFFF) << F.Word3.VJT_INITIAL_OFFSET)
                )
                payloads.extend([w0, w1, w2, w3])
        
        elif mode == 'fold':
            G = Off_Cfg3_V2.Fold
            for neu in neurons:
                w0 = (
                    ((neu.get('fold_range_xy', 0) & 0x7FF) << G.Word0.FOLD_RANGE_XY_OFFSET) |
                    ((neu.get('fold_range_x', 0) & 0x7FF) << G.Word0.FOLD_RANGE_X_OFFSET) |
                    ((neu.get('fold_range_y', 0) & 0x7FF) << G.Word0.FOLD_RANGE_Y_OFFSET) |
                    ((neu.get('fold_skew_xy', 0) & 0x7FF) << G.Word0.FOLD_SKEW_XY_OFFSET) |
                    ((neu.get('fold_skew_x', 0) & 0x7FF) << G.Word0.FOLD_SKEW_X_OFFSET) |
                    ((neu.get('fold_skew_y_hi', 0) & 0x1FF) << G.Word0.FOLD_SKEW_Y_HIGH_OFFSET)
                )
                w1 = (
                    ((neu.get('fold_skew_y_lo', 0) & 0x3) << G.Word1.FOLD_SKEW_Y_LOW_OFFSET) |
                    ((neu.get('fold_axon_xy', 0) & 0x7FF) << G.Word1.FOLD_AXON_XY_OFFSET) |
                    ((neu.get('fold_axon_x', 0) & 0x7FF) << G.Word1.FOLD_AXON_X_OFFSET) |
                    ((neu.get('fold_axon_y', 0) & 0x7FF) << G.Word1.FOLD_AXON_Y_OFFSET) |
                    ((neu.get('fold_number', 0) & 0x1FFFFFFF) << G.Word1.FOLD_NUMBER_OFFSET)
                )
                w2 = (
                    ((neu.get('fold_vjt_3', 0) & 0xFFFFFFFF) << G.Word2.FOLD_VJT_3_OFFSET) |
                    ((neu.get('fold_vjt_2', 0) & 0xFFFFFFFF) << G.Word2.FOLD_VJT_2_OFFSET)
                )
                w3 = (
                    ((neu.get('fold_vjt_1', 0) & 0xFFFFFFFF) << G.Word3.FOLD_VJT_1_OFFSET) |
                    ((neu.get('fold_vjt_0', 0) & 0xFFFFFFFF) << G.Word3.FOLD_VJT_0_OFFSET)
                )
                payloads.extend([w0, w1, w2, w3])
        else:
            raise ValueError(f"Unknown neuron mode: {mode}")

        return OfflineFrameGenV2.gen_packet(
            core_addr, copy_addr, int(FHV2.CONFIG_TYPE3), start_addr, payloads
        )

    @staticmethod
    def gen_input_config(
        core_addr: int,
        copy_addr: int,
        data_blocks: list[list[int]], 
        start_addr: int = 0
    ) -> FrameArrayType:
        """生成 Type 4 (Input SRAM) 配置包。"""
        payloads = []
        for block in data_blocks:
            if len(block) != 8:
                raise ValueError("Each Input SRAM block must have 8 Words (512 bits)")
            payloads.extend(block)
            
        return OfflineFrameGenV2.gen_packet(
            core_addr, copy_addr, int(FHV2.CONFIG_TYPE4), start_addr, payloads
        )

    # =========================================================================
    # 3. 测试帧生成 (Test Request Frames)
    # =========================================================================

    @staticmethod
    def gen_test_request(
        core_addr: int,
        copy_addr: int,
        frame_type: int,
        start_addr: int,
        num_packets: int
    ) -> FrameArrayType:
        """生成测试读取请求 (Test Input Frame)。"""
        if start_addr > 0x1FF:
            raise ValueError(f"start_addr {start_addr} exceeds 9 bits")
        if num_packets > 0x3FFF:
            raise ValueError(f"num_packets {num_packets} exceeds 14 bits")

        # Test Request -> Packet Type Bit [23] = 1
        pkg_type_bit = 1 
        
        header_payload = (
            (pkg_type_bit << FFV2.GENERAL_PACKAGE_TYPE_OFFSET) | 
            ((start_addr & FFV2.GENERAL_PACKAGE_NEU_START_ADDR_MASK) << FFV2.GENERAL_PACKAGE_NEU_START_ADDR_OFFSET) |
            ((num_packets & FFV2.GENERAL_PACKAGE_NUM_MASK) << FFV2.GENERAL_PACKAGE_NUM_OFFSET)
        )
        
        return OfflineFrameGenV2._pack_frame(
            frame_type, core_addr, copy_addr, header_payload
        )

    # =========================================================================
    # 4. 工作帧 & 控制帧 (Single Frames)
    # =========================================================================

    @staticmethod
    def gen_data_frame(
        core_addr: int,
        copy_addr: int,
        timestep: int,
        axon_addr: int,
        data: int,
    ) -> FrameArrayType:
        """生成 WORK_TYPE1 (Data) 帧"""
        F = Off_WF1F_V2
        if timestep > F.TIMESTEP_MASK: raise ValueError("timestep overflow")
        if axon_addr > F.AXON_ADDR_MASK: raise ValueError("axon_addr overflow")
        if data > F.DATA_MASK: raise ValueError("data overflow")

        payload = (
            ((timestep & F.TIMESTEP_MASK) << F.TIMESTEP_OFFSET)
            | ((axon_addr & F.AXON_ADDR_MASK) << F.AXON_ADDR_OFFSET)
            | ((data & F.DATA_MASK) << F.DATA_OFFSET)
        )
        return OfflineFrameGenV2._pack_frame(
            int(FHV2.WORK_TYPE1), core_addr, copy_addr, payload
        )

    @staticmethod
    def gen_vjt_frame(
        core_addr: int,
        copy_addr: int,
        timestep: int,
        axon_addr: int,
        data_part: int,
    ) -> FrameArrayType:
        """生成 WORK_TYPE2 (Vjt) 帧"""
        F = Off_WF2F_V2
        if timestep > F.TIMESTEP_MASK: raise ValueError("timestep overflow")
        if axon_addr > F.AXON_ADDR_MASK: raise ValueError("axon_addr overflow")
        if data_part > F.DATA_PART_MASK: raise ValueError("data_part overflow")

        payload = (
            ((timestep & F.TIMESTEP_MASK) << F.TIMESTEP_OFFSET)
            | ((axon_addr & F.AXON_ADDR_MASK) << F.AXON_ADDR_OFFSET)
            | ((data_part & F.DATA_PART_MASK) << F.DATA_PART_OFFSET)
        )
        return OfflineFrameGenV2._pack_frame(
            int(FHV2.WORK_TYPE2), core_addr, copy_addr, payload
        )

    @staticmethod
    def gen_sync_frame(
        core_addr: int,
        copy_addr: int,
        num_timestep: int,
    ) -> FrameArrayType:
        """生成 CONTROL_TYPE1 (Sync) 帧"""
        F = Off_CF1F
        if num_timestep > F.NUM_TIMESTEP_MASK: raise ValueError("num_timestep overflow")
        payload = (num_timestep & F.NUM_TIMESTEP_MASK) << F.NUM_TIMESTEP_OFFSET
        return OfflineFrameGenV2._pack_frame(
            int(FHV2.CONTROL_TYPE1), core_addr, copy_addr, payload
        )

    @staticmethod
    def gen_init_frame(
        core_addr: int,
        copy_addr: int,
    ) -> FrameArrayType:
        """生成 CONTROL_TYPE2 (Init) 帧"""
        # Init 帧 payload 全为 Reserved (0)
        return OfflineFrameGenV2._pack_frame(
            int(FHV2.CONTROL_TYPE2), core_addr, copy_addr, 0
        )

    @staticmethod
    def gen_complete_frame(
        core_addr: int,
        copy_addr: int,
        pid: int,
    ) -> FrameArrayType:
        """生成 CONTROL_TYPE3 (Complete) 帧"""
        F = Off_CF3F
        if pid > F.PID_MASK: raise ValueError("pid overflow")
        payload = (pid & F.PID_MASK) << F.PID_OFFSET
        return OfflineFrameGenV2._pack_frame(
            int(FHV2.CONTROL_TYPE3), core_addr, copy_addr, payload
        )