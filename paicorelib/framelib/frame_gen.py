from collections.abc import Sequence
from typing import Any, Optional, Union, overload

import numpy as np

from ..coordinate import ChipCoord, Coord
from ..coordinate import ReplicationId as RId
from ..ram_model import OfflineNeuAttrs as OffNeuAttrs
from ..ram_model import OfflineNeuDestInfo as OffNeuDestInfo
from ..ram_model import OnlineNeuAttrs as OnNeuAttrs
from ..ram_model import OnlineNeuDestInfo as OnNeuDestInfo
from ..reg_defs import LCN_EX
from ..reg_defs import WeightWidth as WW
from ..reg_model import OfflineCoreReg as OffCoreReg
from ..reg_model import OnlineCoreReg as OnCoreReg
from .frames import *
from .types import *

__all__ = ["OfflineFrameGen", "OnlineFrameGen"]


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
        core_reg: Union[OffCoreReg, dict[str, Any]],
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
        attrs: Union[OffNeuAttrs, dict[str, Any]],
        dest_info: Union[OffNeuDestInfo, dict[str, Any]],
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
    def gen_magic_init_frame(
        chip_coord: ChipCoord,
        core_coord: Union[Coord, Sequence[Coord]],
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

        magic_frame_cf_1 = []
        magic_frame_cf_2 = []
        magic_frame_wf = []
        init_frame = OfflineWorkFrame4(chip_coord)

        for coord in _core_coord:
            config1 = OfflineConfigFrame1(chip_coord, coord, RId(0, 0), 0)
            work1 = OfflineWorkFrame1(chip_coord, coord, RId(0, 0), 0, 0, 0)

            magic_frame_cf_1.append(config1.value[0])
            if redundant_init:
                magic_frame_cf_1.append(init_frame.value[0])

            magic_frame_cf_2.extend((config1.value[1], config1.value[2]))
            magic_frame_wf.append(work1.value[0])

        if not redundant_init:
            magic_frame_cf_1.append(init_frame.value[0])

        magic_frame_cf_2.extend(magic_frame_wf)

        return np.asarray(magic_frame_cf_1, dtype=FRAME_DTYPE), np.asarray(
            magic_frame_cf_2, dtype=FRAME_DTYPE
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
        core_reg: Union[OffCoreReg, dict[str, Any]],
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
        attrs: Union[OffNeuAttrs, dict[str, Any]],
        dest_info: Union[OffNeuDestInfo, dict[str, Any]],
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
        attrs: Union[OffNeuAttrs, dict[str, Any]],
        dest_info: Union[OffNeuDestInfo, dict[str, Any]],
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
        attrs: Union[OffNeuAttrs, dict[str, Any]],
        dest_info: Union[OffNeuDestInfo, dict[str, Any]],
        lcn_ex: Optional[LCN_EX] = None,
        weight_width: Optional[WW] = None,
        *,
        repeat: Optional[int] = None,
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
        one_input_node: dict[str, Any], data: DataArrayType
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

        indexes = np.nonzero(data.ravel())
        return frame_dest_info[indexes] + data[indexes]

    @staticmethod
    def gen_work_frame2(
        chip_coord: ChipCoord, /, n_sync: IntScalarType
    ) -> OfflineWorkFrame2:
        return OfflineWorkFrame2(chip_coord, int(n_sync))

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
        chip_coord: ChipCoord, core_coord: Coord, rid: RId, /, lut: LUTDataType
    ) -> OnlineConfigFrame1:
        return OnlineConfigFrame1(chip_coord, core_coord, rid, lut)

    @staticmethod
    def gen_config_frame2(
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        /,
        core_reg: Union[OnCoreReg, dict[str, Any]],
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
        attrs: Union[OnNeuAttrs, dict[str, Any]],
        dest_info: Union[OnNeuDestInfo, dict[str, Any]],
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
        test_chip_coord: Coord, core_coord: Coord, rid: RId, /, lut: LUTDataType
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
        core_reg: Union[OnCoreReg, dict[str, Any]],
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
        attrs: Union[OnNeuAttrs, dict[str, Any]],
        dest_info: Union[OnNeuDestInfo, dict[str, Any]],
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
    def gen_work_frame1_1(one_input_node: dict[str, Any]) -> FrameArrayType:
        """Generate the common part of the input spike frames by given the info of one input node.

        Args:
            one_input_node: a dictionary of a single input node that points to online cores.
        """
        return OnlineWorkFrame1_1._frame_dest_reorganized(one_input_node)

    @staticmethod
    def gen_work_frame1_2(
        chip_coord: ChipCoord, coord: Coord, rid: RId = RId(0, 0), /
    ) -> OnlineWorkFrame1_2:
        return OnlineWorkFrame1_2(chip_coord, coord, rid)

    @staticmethod
    def gen_work_frame1_3(
        chip_coord: ChipCoord, coord: Coord, rid: RId = RId(0, 0), /
    ) -> OnlineWorkFrame1_3:
        return OnlineWorkFrame1_3(chip_coord, coord, rid)

    @staticmethod
    def gen_work_frame1_4(
        chip_coord: ChipCoord, coord: Coord, rid: RId = RId(0, 0), /
    ) -> OnlineWorkFrame1_4:
        return OnlineWorkFrame1_4(chip_coord, coord, rid)

    @staticmethod
    def gen_work_frame2(
        chip_coord: ChipCoord, /, n_sync: IntScalarType
    ) -> OnlineWorkFrame2:
        return OnlineWorkFrame2(chip_coord, int(n_sync))

    @staticmethod
    def gen_work_frame3(chip_coord: ChipCoord) -> OnlineWorkFrame3:
        return OnlineWorkFrame3(chip_coord)

    @staticmethod
    def gen_work_frame4(chip_coord: ChipCoord) -> OnlineWorkFrame4:
        return OnlineWorkFrame4(chip_coord)
