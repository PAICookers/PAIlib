from collections.abc import Sequence
from typing import Any, ClassVar, overload,Literal

import numpy as np
from numpy.typing import ArrayLike

from ..coordinate import ChipCoord, Coord
from ..coordinate import ReplicationId as RId
from ..core_defs import LCN_EX, CoreType
from ..core_defs import WeightWidth as WW
from ..core_model import OfflineCoreReg as OffCoreReg
from ..core_model import OnlineCoreReg as OnCoreReg
from ..neuron_model import OfflineNeuAttrs as OffNeuAttrs
from ..neuron_model import OfflineNeuDestInfo as OffNeuDestInfo
from ..neuron_model import OnlineNeuAttrs as OnNeuAttrs
from ..neuron_model import OnlineNeuDestInfo as OnNeuDestInfo
from ..routing_defs import _rid_unset
from .frames import (
    OfflineConfigFrame1,
    OfflineConfigFrame2,
    OfflineConfigFrame3,
    OfflineConfigFrame4,
    OfflineTestInFrame1,
    OfflineTestInFrame2,
    OfflineTestInFrame3,
    OfflineTestInFrame4,
    OfflineTestOutFrame1,
    OfflineTestOutFrame2,
    OfflineTestOutFrame3,
    OfflineTestOutFrame4,
    OfflineWorkFrame1,
    OfflineWorkFrame2,
    OfflineWorkFrame3,
    OfflineWorkFrame4,
    OnlineConfigFrame1,
    OnlineConfigFrame2,
    OnlineConfigFrame3,
    OnlineConfigFrame4,
    OnlineTestInFrame1,
    OnlineTestInFrame2,
    OnlineTestInFrame3,
    OnlineTestInFrame4,
    OnlineTestOutFrame1,
    OnlineTestOutFrame2,
    OnlineTestOutFrame3,
    OnlineTestOutFrame4,
    OnlineWorkFrame1_1,
    OnlineWorkFrame1_2,
    OnlineWorkFrame1_3,
    OnlineWorkFrame1_4,
    OnlineWorkFrame2,
    OnlineWorkFrame3,
    OnlineWorkFrame4,
)
from .types import (
    FRAME_DTYPE,
    PAYLOAD_DATA_DTYPE,
    FrameArrayType,
    IntScalarType,
    LUTDataType,
    PayloadDataType,
)

__all__ = ["OfflineFrameGen", "OnlineFrameGen", "ChipFrameGen", "OfflineFrameGenV2"]



from .frame_defs import (
    FrameFormatV2 as FFV2,
    FrameHeaderV2 as FHV2,
    FramePackageType,
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

    @staticmethod
    def _pack_frame(
        header: int,
        core_addr: int,
        copy_addr: int,
        payload: int,
    ) -> FrameArrayType:

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

        frames = []
        n_package = len(payload_data)
        
        if start_addr > 0x1FF:
            raise ValueError(f"start_addr {hex(start_addr)} exceeds 9 bits")
        if n_package > 0x3FFF:
            raise ValueError(f"Number of packages {n_package} exceeds 14 bits")
            
        # V2.5 Protocol: [23] Package Type (0: Config/TestOut, 1: TestIn)
        pkg_type_bit = int(FramePackageType.TESTIN) if is_test_request else int(FramePackageType.CONF_TESTOUT)
        
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


    @staticmethod
    def gen_core_config(
        core_addr: int,
        copy_addr: int,
        params: dict[str, int]
    ) -> FrameArrayType:

        F = Off_Cfg1_V2
        
        # Word 1
        w1 = (
            ((params['snn_ann'] & F.Word1.SNN_ANN_MASK) << F.Word1.SNN_ANN_OFFSET) |
            ((params['max_pooling'] & F.Word1.MAX_POOLING_MASK) << F.Word1.MAX_POOLING_OFFSET) |
            ((params['add_potential'] & F.Word1.ADD_POTENTIAL_MASK) << F.Word1.ADD_POTENTIAL_OFFSET) |
            ((params['zero_output'] & F.Word1.ZERO_OUTPUT_MASK) << F.Word1.ZERO_OUTPUT_OFFSET) |
            ((params['input_sign'] & F.Word1.INPUT_SIGN_MASK) << F.Word1.INPUT_SIGN_OFFSET) |
            ((params['input_width'] & F.Word1.INPUT_WIDTH_MASK) << F.Word1.INPUT_WIDTH_OFFSET) |
            ((params['output_sign'] & F.Word1.OUTPUT_SIGN_MASK) << F.Word1.OUTPUT_SIGN_OFFSET) |
            ((params['output_width'] & F.Word1.OUTPUT_WIDTH_MASK) << F.Word1.OUTPUT_WIDTH_OFFSET) |
            ((params['weight_sign'] & F.Word1.WEIGHT_SIGN_MASK) << F.Word1.WEIGHT_SIGN_OFFSET) |
            ((params['weight_width'] & F.Word1.WEIGHT_WIDTH_MASK) << F.Word1.WEIGHT_WIDTH_OFFSET) |
            ((params['lcn'] & F.Word1.LCN_MASK) << F.Word1.LCN_OFFSET) |
            ((params['target_lcn'] & F.Word1.TARGET_LCN_MASK) << F.Word1.TARGET_LCN_OFFSET) |
            ((params['axon_skew'] & F.Word1.AXON_SKEW_MASK) << F.Word1.AXON_SKEW_OFFSET) |
            ((params['neuron_number'] & F.Word1.NEURON_NUMBER_MASK) << F.Word1.NEURON_NUMBER_OFFSET) |
            ((params['test_core_xy'] & F.Word1.TEST_CORE_XY_MASK) << F.Word1.TEST_CORE_XY_OFFSET) |
            ((params['test_core_x'] & F.Word1.TEST_CORE_X_MASK) << F.Word1.TEST_CORE_X_OFFSET) |
            ((params['test_core_y_high'] & F.Word1.TEST_CORE_Y_HIGH2_MASK) << F.Word1.TEST_CORE_Y_HIGH2_OFFSET)
        )

        # Word 2
        w2 = (
            ((params['test_core_y_low'] & F.Word2.TEST_CORE_Y_LOW4_MASK) << F.Word2.TEST_CORE_Y_LOW4_OFFSET) |
            ((params['global_send'] & F.Word2.GLOBAL_SEND_MASK) << F.Word2.GLOBAL_SEND_OFFSET) |
            ((params['csc_accelerate'] & F.Word2.CSC_ACCELERATE_MASK) << F.Word2.CSC_ACCELERATE_OFFSET) |
            ((params['global_receive'] & F.Word2.GLOBAL_RECEIVE_MASK) << F.Word2.GLOBAL_RECEIVE_OFFSET) |
            ((params['thread_number'] & F.Word2.THREAD_NUMBER_MASK) << F.Word2.THREAD_NUMBER_OFFSET) |
            ((params['busy_cycle'] & F.Word2.BUSY_CYCLE_MASK) << F.Word2.BUSY_CYCLE_OFFSET) |
            ((params['delay_cycle'] & F.Word2.DELAY_CYCLE_MASK) << F.Word2.DELAY_CYCLE_OFFSET) |
            ((params['width_cycle'] & F.Word2.WIDTH_CYCLE_MASK) << F.Word2.WIDTH_CYCLE_OFFSET)
        )

        # Word 3
        w3 = (
            ((params['tick_start'] & F.Word3.TICK_START_MASK) << F.Word3.TICK_START_OFFSET) |
            ((params['tick_duration'] & F.Word3.TICK_DURATION_MASK) << F.Word3.TICK_DURATION_OFFSET) |
            ((params['tick_initial'] & F.Word3.TICK_INITIAL_MASK) << F.Word3.TICK_INITIAL_OFFSET)
        )

        return OfflineFrameGenV2.gen_packet(
            core_addr, copy_addr, int(FHV2.CONFIG_TYPE1), 0, [w1, w2, w3]
        )

    staticmethod
    def gen_lut_config(
        core_addr: int,
        copy_addr: int,
        potentials: list[int] | np.ndarray,   # 支持 list 或 numpy array
        activations: list[int] | np.ndarray,  # 支持 list 或 numpy array
        start_index: int = 0
    ) -> FrameArrayType:

        if len(potentials) != len(activations):
            raise ValueError(f"LUT size mismatch: potentials has {len(potentials)} entries, "
                             f"but activations has {len(activations)} entries.")

        F = Off_Cfg2_V2
        payloads = []
        
        for pot, act in zip(potentials, activations):
            w = (
                ((int(pot) & F.POTENTIAL_MASK)  << F.POTENTIAL_OFFSET) |
                ((int(act) & F.ACTIVATION_MASK) << F.ACTIVATION_OFFSET)
            )
            payloads.append(w)
            
        return OfflineFrameGenV2.gen_packet(
            core_addr, copy_addr, int(FHV2.CONFIG_TYPE2), start_index, payloads
        )

    @staticmethod
    def gen_neuron_config(
        core_addr: int,
        copy_addr: int,
        common_attrs: dict[str, int],    # 神经元属性（阈值、Leak、Reset模式等）
        specific_targets: list[dict[str, int]], # 神经元地址与时序（addr_*, tick_relative）
        start_addr: int = 0,
        mode: Literal["full", "fold"] = "full"
    ) -> FrameArrayType:

        payloads = []
        
        if mode == "full":
            F = Off_Cfg3_V2.Full
            for target in specific_targets:

                neu = {**common_attrs, **target}

                w1 = (
                    ((neu['tick_relative']   & F.Word1.TICK_RELATIVE_MASK)   << F.Word1.TICK_RELATIVE_OFFSET) |
                    ((neu['addr_axon']       & F.Word1.ADDR_AXON_MASK)       << F.Word1.ADDR_AXON_OFFSET) |
                    ((neu['addr_core_xy']    & F.Word1.ADDR_CORE_XY_MASK)    << F.Word1.ADDR_CORE_XY_OFFSET) |
                    ((neu['addr_core_x']     & F.Word1.ADDR_CORE_X_MASK)     << F.Word1.ADDR_CORE_X_OFFSET) |
                    ((neu['addr_core_y']     & F.Word1.ADDR_CORE_Y_MASK)     << F.Word1.ADDR_CORE_Y_OFFSET) |
                    ((neu['addr_copy_xy']    & F.Word1.ADDR_COPY_XY_MASK)    << F.Word1.ADDR_COPY_XY_OFFSET) |
                    ((neu['addr_copy_x']     & F.Word1.ADDR_COPY_X_MASK)     << F.Word1.ADDR_COPY_X_OFFSET) |
                    ((neu['addr_copy_y']     & F.Word1.ADDR_COPY_Y_MASK)     << F.Word1.ADDR_COPY_Y_OFFSET) |
                    ((neu['weight_skew_high']& F.Word1.WEIGHT_SKEW_HIGH_MASK)<< F.Word1.WEIGHT_SKEW_HIGH_OFFSET)
                )
                w2 = (
                    ((neu['weight_skew_low']     & F.Word2.WEIGHT_SKEW_LOW_MASK)        << F.Word2.WEIGHT_SKEW_LOW_OFFSET) |
                    ((neu['weight_addr_start']   & F.Word2.WEIGHT_ADDRESS_START_MASK)   << F.Word2.WEIGHT_ADDRESS_START_OFFSET) |
                    ((neu['weight_addr_end']     & F.Word2.WEIGHT_ADDRESS_END_MASK)     << F.Word2.WEIGHT_ADDRESS_END_OFFSET) |
                    ((neu['output_type']         & F.Word2.OUTPUT_TYPE_MASK)            << F.Word2.OUTPUT_TYPE_OFFSET) |
                    ((neu['fold_type']           & F.Word2.FOLD_TYPE_MASK)              << F.Word2.FOLD_TYPE_OFFSET) |
                    ((neu['neuron_type']         & F.Word2.NEURON_TYPE_MASK)            << F.Word2.NEURON_TYPE_OFFSET) |
                    ((neu['vjt']                 & F.Word2.VJT_MASK)                    << F.Word2.VJT_OFFSET)
                )

                w3 = (
                    ((neu['reset_mode']      & F.Word3.RESET_MODE_MASK)          << F.Word3.RESET_MODE_OFFSET) |
                    ((neu['reset_v']         & F.Word3.RESET_V_MASK)             << F.Word3.RESET_V_OFFSET) |
                    ((neu['thres_neg_mode']  & F.Word3.THRESHOLD_NEG_MODE_MASK)  << F.Word3.THRESHOLD_NEG_MODE_OFFSET) |
                    ((neu['thres_pos_mode']  & F.Word3.THRESHOLD_POS_MODE_MASK)  << F.Word3.THRESHOLD_POS_MODE_OFFSET) |
                    ((neu['thres_neg']       & F.Word3.THRESHOLD_NEG_MASK)       << F.Word3.THRESHOLD_NEG_OFFSET) |
                    ((neu['thres_pos_hi']    & F.Word3.THRESHOLD_POS_HIGH_MASK)    << F.Word3.THRESHOLD_POS_HIGH_OFFSET)
                )
                w4 = (
                    ((neu['thres_pos_low']    & F.Word4.THRESHOLD_POS_LOW_MASK)      << F.Word4.THRESHOLD_POS_LOW_OFFSET) |
                    ((neu['lateral_inhibit'] & F.Word4.LATERAL_INHIBITION_MASK)    << F.Word4.LATERAL_INHIBITION_OFFSET) |
                    ((neu['leak_multi_seq']  & F.Word4.LEAK_MULTI_SEQUENCE_MASK)   << F.Word4.LEAK_MULTI_SEQUENCE_OFFSET) |
                    ((neu['leak_multi_in']   & F.Word4.LEAK_MULTI_INPUT_MASK)      << F.Word4.LEAK_MULTI_INPUT_OFFSET) |
                    ((neu['leak_multi_mode'] & F.Word4.LEAK_MULTI_MODE_MASK)       << F.Word4.LEAK_MULTI_MODE_OFFSET) |
                    ((neu['leak_add_mode']   & F.Word4.LEAK_ADD_MODE_MASK)         << F.Word4.LEAK_ADD_MODE_OFFSET) |
                    ((neu['leak_tau']        & F.Word4.LEAK_TAU_MASK)              << F.Word4.LEAK_TAU_OFFSET) |
                    ((neu['leak_v']          & F.Word4.LEAK_V_MASK)                << F.Word4.LEAK_V_OFFSET) |
                    ((neu['weight_compress'] & F.Word4.WEIGHT_COMPRESS_MASK)       << F.Word4.WEIGHT_COMPRESS_OFFSET) |
                    ((neu['vjt_initial']     & F.Word4.VJT_INITIAL_MASK)           << F.Word4.VJT_INITIAL_OFFSET)
                )
                payloads.extend([w1, w2, w3, w4])
        
        elif mode == "fold":
            G = Off_Cfg3_V2.Fold
            for target in specific_targets:
                neu = {**common_attrs, **target}

                w1 = (
                    ((neu['fold_range_xy']   & G.Word1.FOLD_RANGE_XY_MASK)     << G.Word1.FOLD_RANGE_XY_OFFSET) |
                    ((neu['fold_range_x']    & G.Word1.FOLD_RANGE_X_MASK)      << G.Word1.FOLD_RANGE_X_OFFSET) |
                    ((neu['fold_range_y']    & G.Word1.FOLD_RANGE_Y_MASK)      << G.Word1.FOLD_RANGE_Y_OFFSET) |
                    ((neu['fold_skew_xy']    & G.Word1.FOLD_SKEW_XY_MASK)      << G.Word1.FOLD_SKEW_XY_OFFSET) |
                    ((neu['fold_skew_x']     & G.Word1.FOLD_SKEW_X_MASK)       << G.Word1.FOLD_SKEW_X_OFFSET) |
                    ((neu['fold_skew_y_hi']  & G.Word1.FOLD_SKEW_Y_HIGH_MASK)  << G.Word1.FOLD_SKEW_Y_HIGH_OFFSET)
                )

                w2 = (
                    ((neu['fold_skew_y_low']  & G.Word2.FOLD_SKEW_Y_LOW_MASK)   << G.Word2.FOLD_SKEW_Y_LOW_OFFSET) |
                    ((neu['fold_axon_xy']    & G.Word2.FOLD_AXON_XY_MASK)      << G.Word2.FOLD_AXON_XY_OFFSET) |
                    ((neu['fold_axon_x']     & G.Word2.FOLD_AXON_X_MASK)       << G.Word2.FOLD_AXON_X_OFFSET) |
                    ((neu['fold_axon_y']     & G.Word2.FOLD_AXON_Y_MASK)       << G.Word2.FOLD_AXON_Y_OFFSET) |
                    ((neu['fold_number']     & G.Word2.FOLD_NUMBER_MASK)       << G.Word2.FOLD_NUMBER_OFFSET)
                )

                w3 = (
                    ((neu['fold_vjt_3']      & G.Word3.FOLD_VJT_3_MASK)        << G.Word3.FOLD_VJT_3_OFFSET) |
                    ((neu['fold_vjt_2']      & G.Word3.FOLD_VJT_2_MASK)        << G.Word3.FOLD_VJT_2_OFFSET)
                )

                w4 = (
                    ((neu['fold_vjt_1']      & G.Word4.FOLD_VJT_1_MASK)        << G.Word4.FOLD_VJT_1_OFFSET) |
                    ((neu['fold_vjt_0']      & G.Word4.FOLD_VJT_0_MASK)        << G.Word4.FOLD_VJT_0_OFFSET)
                )
                payloads.extend([w1, w2, w3, w4])
        
        else:
            raise ValueError(f"Unknown neuron mode: {mode}")

        return OfflineFrameGenV2.gen_packet(
            core_addr, copy_addr, int(FHV2.CONFIG_TYPE3), start_addr, payloads
        )


    @staticmethod
    def gen_test_request(
        core_addr: int,
        copy_addr: int,
        frame_type: int,
        start_addr: int,
        num_packets: int
    ) -> FrameArrayType:

        if start_addr > 0x1FF:
            raise ValueError(f"start_addr {start_addr} exceeds 9 bits")
        if num_packets > 0x3FFF:
            raise ValueError(f"num_packets {num_packets} exceeds 14 bits")

        # Test Request -> Packet Type Bit [23] = 1 (TESTIN)
        pkg_type_bit = int(FramePackageType.TESTIN)
        
        header_payload = (
            (pkg_type_bit << FFV2.GENERAL_PACKAGE_TYPE_OFFSET) | 
            ((start_addr & FFV2.GENERAL_PACKAGE_NEU_START_ADDR_MASK) << FFV2.GENERAL_PACKAGE_NEU_START_ADDR_OFFSET) |
            ((num_packets & FFV2.GENERAL_PACKAGE_NUM_MASK) << FFV2.GENERAL_PACKAGE_NUM_OFFSET)
        )
        
        return OfflineFrameGenV2._pack_frame(
            frame_type, core_addr, copy_addr, header_payload
        )

    @staticmethod
    def gen_data_frame(
        core_addr: int,
        copy_addr: int,
        timestep: int,
        axon_addr: int,
        data: int,
        time_width: int,
        axon_width: int, 
        data_width: int   
    ) -> FrameArrayType:

        if time_width + axon_width != 17:
            raise ValueError(f"Invalid bit widths: time_width({time_width}) + axon_width({axon_width}) must be 17.")

        F = Off_WF1F_V2

        dynamic_time_mask = (1 << time_width) - 1
        dynamic_axon_mask = (1 << axon_width) - 1
        dynamic_data_mask = (1 << data_width) - 1

        if timestep > dynamic_time_mask: 
            raise ValueError(f"timestep overflow: value {timestep} exceeds {time_width}-bit width")
        if axon_addr > dynamic_axon_mask: 
            raise ValueError(f"axon_addr overflow: value {axon_addr} exceeds {axon_width}-bit width")
        if data > dynamic_data_mask: 
            raise ValueError(f"data overflow: value {data} exceeds {data_width}-bit width")

        mixed_addr = (timestep << axon_width) | axon_addr

        addr_msb = (mixed_addr >> 16) & 0x1
        addr_low = mixed_addr & 0xFFFF

        payload = (
            (addr_msb << F.TIMESTEP_HIGH_OFFSET) |
            (addr_low << F.AXON_ADDR_OFFSET) |
            ((data & dynamic_data_mask) << F.DATA_OFFSET)
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
        vjt: int,
        time_width: int, 
        axon_width: int, 
        vjt_width: int   
    ) -> FrameArrayType:

        if time_width + axon_width != 17:
            raise ValueError(f"Invalid bit widths: time_width({time_width}) + axon_width({axon_width}) must be 17.")

        F = Off_WF2F_V2

        dynamic_time_mask = (1 << time_width) - 1
        dynamic_axon_mask = (1 << axon_width) - 1
        dynamic_vjt_mask  = (1 << vjt_width) - 1

        if timestep > dynamic_time_mask: 
            raise ValueError(f"timestep overflow: {timestep}")
        if axon_addr > dynamic_axon_mask: 
            raise ValueError(f"axon_addr overflow: {axon_addr}")
        if vjt > dynamic_vjt_mask: 
            raise ValueError(f"vjt overflow: {vjt}")

        mixed_addr = (timestep << axon_width) | axon_addr
        addr_msb = (mixed_addr >> 16) & 0x1
        addr_low = mixed_addr & 0xFFFF

        payload = (
            (addr_msb << F.TIMESTEP_HIGH_OFFSET) |
            (addr_low << F.AXON_ADDR_OFFSET) | 
            ((vjt & dynamic_vjt_mask) << F.VJT_OFFSET)
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

        return OfflineFrameGenV2._pack_frame(
            int(FHV2.CONTROL_TYPE2), core_addr, copy_addr, 0
        )

    @staticmethod
    def gen_complete_frame(
        core_addr: int,
        copy_addr: int,
        thread_id: int,
    ) -> FrameArrayType:
        
        F = Off_CF3F
        if thread_id > F.THREAD_ID_MASK: raise ValueError("thread_id overflow")
        payload = (thread_id & F.THREAD_ID_MASK) << F.THREAD_ID_OFFSET
        return OfflineFrameGenV2._pack_frame(
            int(FHV2.CONTROL_TYPE3), core_addr, copy_addr, payload
        )