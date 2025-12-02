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

<<<<<<< HEAD
__all__ = ["OfflineFrameGen", "OnlineFrameGen", "ChipFrameGen"]
=======
from .frame_defs import (
    FrameFormatV2 as FFV2,
    FrameHeaderV2 as FHV2,
    OfflineWorkFrame1FormatV2 as Off_WF1F_V2,
    OfflineWorkFrame2FormatV2 as Off_WF2F_V2,
    OfflineControlFrame1Format as Off_CF1F,
    OfflineControlFrame2Format as Off_CF2F,
    OfflineControlFrame3Format as Off_CF3F,
)


__all__ = ["OfflineFrameGen", "OnlineFrameGen"]
>>>>>>> e5acec3 (feat: 添加 V2 版本的在线配置帧格式类，扩展帧生成器功能)


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
<<<<<<< HEAD


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
=======
class OfflineFrameGenV2:
    """Offline frame generator for FFV2 format.

    使用 FFV2 定义的:
    - WORK_TYPE1: 数据帧 (OfflineWorkFrame1FormatV2)
    - WORK_TYPE2: Vjt 帧 (OfflineWorkFrame2FormatV2)
    - CONTROL_TYPE1: Sync 控制帧
    - CONTROL_TYPE2: Init 控制帧
    - CONTROL_TYPE3: Complete 控制帧
    """

    # ----------- 工具函数：组装 64-bit 帧 -----------

    @staticmethod
    def _pack_frame(
        header: int,
        core_addr: int,
        copy_addr: int,
        payload: int,
    ) -> FrameArrayType:
        """按 FFV2 布局打包单个帧，返回 (1,) 的 FrameArrayType。"""
        frame = (
            ((header & FFV2.GENERAL_HEADER_MASK) << FFV2.GENERAL_HEADER_OFFSET)
            | ((core_addr & FFV2.GENERAL_CORE_ADDR_MASK) << FFV2.GENERAL_CORE_ADDR_OFFSET)
            | ((copy_addr & FFV2.GENERAL_COPY_ADDR_MASK) << FFV2.GENERAL_COPY_ADDR_OFFSET)
            | (payload & FFV2.GENERAL_PAYLOAD_MASK)
        )
        return np.asarray([FRAME_DTYPE(frame)], dtype=FRAME_DTYPE)

    # ----------- WORK_TYPE1: Data 帧 -----------

    @staticmethod
    def gen_data_frame(
        core_addr: int,
        copy_addr: int,
        /,
        timestep: IntScalarType,
        axon_addr: IntScalarType,
        data: DataType,
    ) -> FrameArrayType:
        """生成 FFV2 的 data work frame (WORK_TYPE1).

        payload 布局 (OfflineWorkFrame1FormatV2):
            [23:17] timestep (7bit)
            [16:8]  axon_addr (9bit)
            [7:0]   data (8bit)
        """
        ts = int(timestep)
        ax = int(axon_addr)
        d = int(data)

        # 简单范围检查（超过掩码直接报错，和 Frame 里的做法类似）:contentReference[oaicite:5]{index=5}
        if ts < 0 or ts > Off_WF1F_V2.TIMESTEP_MASK:
            raise ValueError(
                f"timestep out of range [0, {Off_WF1F_V2.TIMESTEP_MASK}], got {ts}."
            )
        if ax < 0 or ax > Off_WF1F_V2.AXON_ADDR_MASK:
            raise ValueError(
                f"axon_addr out of range [0, {Off_WF1F_V2.AXON_ADDR_MASK}], got {ax}."
            )
        if d < 0 or d > Off_WF1F_V2.DATA_MASK:
            # 这里假定 data 为无符号 8bit，如需有符号可以自己改成 np.int8 范围
            raise ValueError(
                f"data out of range [0, {Off_WF1F_V2.DATA_MASK}], got {d}."
            )

        payload = (
            ((ts & Off_WF1F_V2.TIMESTEP_MASK) << Off_WF1F_V2.TIMESTEP_OFFSET)
            | ((ax & Off_WF1F_V2.AXON_ADDR_MASK) << Off_WF1F_V2.AXON_ADDR_OFFSET)
            | ((d & Off_WF1F_V2.DATA_MASK) << Off_WF1F_V2.DATA_OFFSET)
        )

        return OfflineFrameGenV2._pack_frame(
            int(FHV2.WORK_TYPE1), core_addr, copy_addr, payload
        )

    # ----------- WORK_TYPE2: Vjt 帧 -----------

    @staticmethod
    def gen_vjt_frame(
        core_addr: int,
        copy_addr: int,
        /,
        timestep: IntScalarType,
        axon_addr: IntScalarType,
        data_part: IntScalarType,
    ) -> FrameArrayType:
        """生成 FFV2 的 Vjt work frame (WORK_TYPE2).

        payload 布局 (OfflineWorkFrame2FormatV2):
            [23:17] timestep (7bit)
            [16:8]  axon_addr (9bit)
            [7:0]   data_part (8bit, Vjt 的一部分)
        """
        ts = int(timestep)
        ax = int(axon_addr)
        dp = int(data_part)

        if ts < 0 or ts > Off_WF2F_V2.TIMESTEP_MASK:
            raise ValueError(
                f"timestep out of range [0, {Off_WF2F_V2.TIMESTEP_MASK}], got {ts}."
            )
        if ax < 0 or ax > Off_WF2F_V2.AXON_ADDR_MASK:
            raise ValueError(
                f"axon_addr out of range [0, {Off_WF2F_V2.AXON_ADDR_MASK}], got {ax}."
            )
        if dp < 0 or dp > Off_WF2F_V2.DATA_PART_MASK:
            raise ValueError(
                f"data_part out of range [0, {Off_WF2F_V2.DATA_PART_MASK}], got {dp}."
            )

        payload = (
            ((ts & Off_WF2F_V2.TIMESTEP_MASK) << Off_WF2F_V2.TIMESTEP_OFFSET)
            | ((ax & Off_WF2F_V2.AXON_ADDR_MASK) << Off_WF2F_V2.AXON_ADDR_OFFSET)
            | ((dp & Off_WF2F_V2.DATA_PART_MASK) << Off_WF2F_V2.DATA_PART_OFFSET)
        )

        return OfflineFrameGenV2._pack_frame(
            int(FHV2.WORK_TYPE2), core_addr, copy_addr, payload
        )

    # ----------- CONTROL_TYPE1: Sync 控制帧 -----------

    @staticmethod
    def gen_sync_frame(
        core_addr: int,
        copy_addr: int,
        /,
        num_timestep: IntScalarType,
    ) -> FrameArrayType:
        """生成 FFV2 的 Sync 控制帧 (CONTROL_TYPE1).

        payload 布局 (OfflineControlFrame1Format):
            [23:0] num_timestep
        """
        n_ts = int(num_timestep)
        if n_ts < 0 or n_ts > Off_CF1F.NUM_TIMESTEP_MASK:
            raise ValueError(
                f"num_timestep out of range [0, {Off_CF1F.NUM_TIMESTEP_MASK}], got {n_ts}."
            )

        payload = (
            (n_ts & Off_CF1F.NUM_TIMESTEP_MASK) << Off_CF1F.NUM_TIMESTEP_OFFSET
        )

        return OfflineFrameGenV2._pack_frame(
            int(FHV2.CONTROL_TYPE1), core_addr, copy_addr, payload
        )

    # ----------- CONTROL_TYPE2: Init 控制帧 -----------

    @staticmethod
    def gen_init_frame(
        core_addr: int,
        copy_addr: int,
    ) -> FrameArrayType:
        """生成 FFV2 的 Init 控制帧 (CONTROL_TYPE2).

        OfflineControlFrame2Format 只有保留位，全 0 即可。:contentReference[oaicite:6]{index=6}
        """
        payload = 0  # RESERVED 全 0
        return OfflineFrameGenV2._pack_frame(
            int(FHV2.CONTROL_TYPE2), core_addr, copy_addr, payload
        )

    # ----------- CONTROL_TYPE3: Complete 控制帧 -----------

    @staticmethod
    def gen_complete_frame(
        core_addr: int,
        copy_addr: int,
        /,
        pid: IntScalarType,
    ) -> FrameArrayType:
        """生成 FFV2 的 Complete 控制帧 (CONTROL_TYPE3).

        payload 布局 (OfflineControlFrame3Format):
            [23:0] PID
        """
        p = int(pid)
        if p < 0 or p > Off_CF3F.PID_MASK:
            raise ValueError(
                f"pid out of range [0, {Off_CF3F.PID_MASK}], got {p}."
            )

        payload = (p & Off_CF3F.PID_MASK) << Off_CF3F.PID_OFFSET

        return OfflineFrameGenV2._pack_frame(
            int(FHV2.CONTROL_TYPE3), core_addr, copy_addr, payload
        )

>>>>>>> e5acec3 (feat: 添加 V2 版本的在线配置帧格式类，扩展帧生成器功能)
