from collections.abc import Sequence
from typing import Any, Union, overload

import numpy as np
from numpy.typing import NDArray

from paicorelib import LCN_EX, Coord, NeuronAttrs, NeuronDestInfo, ParamsReg
from paicorelib import ReplicationId as RId
from paicorelib import WeightPrecision as WP

from .frames import *
from .types import FRAME_DTYPE, DataArrayType, FrameArrayType, IntScalarType

__all__ = ["OfflineFrameGen"]


class OfflineFrameGen:
    """Offline frame generator."""

    @staticmethod
    def gen_config_frame1(
        chip_coord: Coord, core_coord: Coord, rid: RId, /, random_seed: IntScalarType
    ) -> OfflineConfigFrame1:
        return OfflineConfigFrame1(chip_coord, core_coord, rid, int(random_seed))

    @overload
    @staticmethod
    def gen_config_frame2(
        chip_coord: Coord, core_coord: Coord, rid: RId, /, params_reg: ParamsReg
    ) -> OfflineConfigFrame2: ...

    @overload
    @staticmethod
    def gen_config_frame2(
        chip_coord: Coord, core_coord: Coord, rid: RId, /, params_reg: dict[str, Any]
    ) -> OfflineConfigFrame2: ...

    @staticmethod
    def gen_config_frame2(
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        params_reg: Union[ParamsReg, dict[str, Any]],
    ) -> OfflineConfigFrame2:
        if isinstance(params_reg, ParamsReg):
            _params_reg = params_reg.model_dump(by_alias=True)
        else:
            _params_reg = params_reg

        return OfflineConfigFrame2(chip_coord, core_coord, rid, _params_reg)

    @overload
    @staticmethod
    def gen_config_frame3(
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        sram_start_addr: int,
        n_neuron: int,
        attrs: NeuronAttrs,
        dest_info: NeuronDestInfo,
        lcn_ex: LCN_EX,
        weight_precision: WP,
    ) -> OfflineConfigFrame3: ...

    @overload
    @staticmethod
    def gen_config_frame3(
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        sram_start_addr: int,
        n_neuron: int,
        attrs: dict[str, Any],
        dest_info: dict[str, Any],
        lcn_ex: LCN_EX,
        weight_precision: WP,
    ) -> OfflineConfigFrame3: ...

    @staticmethod
    def gen_config_frame3(
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        sram_start_addr: int,
        n_neuron: int,
        attrs: Union[NeuronAttrs, dict[str, Any]],
        dest_info: Union[NeuronDestInfo, dict[str, Any]],
        lcn_ex: LCN_EX,
        weight_precision: WP,
    ) -> OfflineConfigFrame3:
        if isinstance(attrs, NeuronAttrs):
            _attrs = attrs.model_dump(by_alias=True)
        else:
            _attrs = attrs

        if isinstance(dest_info, NeuronDestInfo):
            _dest_info = dest_info.model_dump(by_alias=True)
        else:
            _dest_info = dest_info

        return OfflineConfigFrame3(
            chip_coord,
            core_coord,
            rid,
            sram_start_addr,
            n_neuron,
            _attrs,
            _dest_info,
            repeat=(1 << lcn_ex) * (1 << weight_precision),
        )

    @staticmethod
    def gen_config_frame4(
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        sram_start_addr: int,
        n_data_package: int,
        weight_ram: FrameArrayType,
    ) -> OfflineConfigFrame4:
        return OfflineConfigFrame4(
            chip_coord, core_coord, rid, sram_start_addr, n_data_package, weight_ram
        )

    @staticmethod
    def gen_magic_init_frame(
        chip_coord: Coord,
        core_coord: Union[Coord, Sequence[Coord]],
        redundant_init: bool = True,
    ) -> Union[FrameArrayType, tuple[FrameArrayType, FrameArrayType]]:
        """Magic initialization frames for PAICORE 2.0. DO NOT MODIFY!

        Args:
            - chip_coord: coordinate of the target chip.
            - core_coord: coordinates of the target cores.
            - redundant_init: whether to use redundant initialization frames, in case of failure.

        If use redundant initialization frames, the magic frames are composed of:
            1. [config1[0] of core #1] + [init frame] + [config1[0] of core #2] + [init frame] + ...\
                + [config1[0] of core #N] + [init frame]

            2. [config1[1] of core #1] + [config1[1] of core #2] + [config1[1] of core #2] + ... +  \
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
        chip_coord: Coord, core_coord: Coord, rid: RId, /
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
        chip_coord: Coord, core_coord: Coord, rid: RId, /
    ) -> OfflineTestInFrame2:
        return OfflineTestInFrame2(chip_coord, core_coord, rid)

    @overload
    @staticmethod
    def gen_testout_frame2(
        test_chip_coord: Coord, core_coord: Coord, rid: RId, /, params_reg: ParamsReg
    ) -> OfflineTestOutFrame2: ...

    @overload
    @staticmethod
    def gen_testout_frame2(
        test_chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        params_reg: dict[str, Any],
    ) -> OfflineTestOutFrame2: ...

    @staticmethod
    def gen_testout_frame2(
        test_chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        params_reg: Union[ParamsReg, dict[str, Any]],
    ) -> OfflineTestOutFrame2:
        if isinstance(params_reg, ParamsReg):
            pr = params_reg.model_dump(by_alias=True)
        else:
            pr = params_reg

        return OfflineTestOutFrame2(test_chip_coord, core_coord, rid, pr)

    @staticmethod
    def gen_testin_frame3(
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        sram_start_addr: int,
        data_package_num: int,
    ) -> OfflineTestInFrame3:
        return OfflineTestInFrame3(
            chip_coord, core_coord, rid, sram_start_addr, data_package_num
        )

    @staticmethod
    def gen_testout_frame3(
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        sram_start_addr: int,
        n_neuron: int,
        attrs: Union[NeuronAttrs, dict[str, Any]],
        dest_info: Union[NeuronDestInfo, dict[str, Any]],
        *,
        lcn_ex: LCN_EX = LCN_EX.LCN_1X,
        weight_precision: WP = WP.WEIGHT_WIDTH_1BIT,
    ) -> OfflineTestOutFrame3:
        if isinstance(attrs, NeuronAttrs):
            _attrs = attrs.model_dump(by_alias=True)
        else:
            _attrs = attrs

        if isinstance(dest_info, NeuronDestInfo):
            _dest_info = dest_info.model_dump(by_alias=True)
        else:
            _dest_info = dest_info

        return OfflineTestOutFrame3(
            chip_coord,
            core_coord,
            rid,
            sram_start_addr,
            n_neuron,
            _attrs,
            _dest_info,
            repeat=(1 << lcn_ex) * (1 << weight_precision),
        )

    @staticmethod
    def gen_testin_frame4(
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        sram_start_addr: int,
        data_package_num: int,
    ) -> OfflineTestInFrame4:
        return OfflineTestInFrame4(
            chip_coord, core_coord, rid, sram_start_addr, data_package_num
        )

    @staticmethod
    def gen_testout_frame4(
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        sram_start_addr: int,
        n_data_package: int,
        weight_ram: FrameArrayType,
    ) -> OfflineTestOutFrame4:
        return OfflineTestOutFrame4(
            chip_coord, core_coord, rid, sram_start_addr, n_data_package, weight_ram
        )

    @staticmethod
    def gen_work_frame1(
        one_input_node: dict[str, Any],
        data: DataArrayType,
    ) -> FrameArrayType:
        """Generate the common part of the input spike frames by given the info \
            of one input node.

        Args:
            - one_input_node: a dictionary of one input node.
        """
        common_frame_dest = OfflineWorkFrame1._frame_dest_reorganized(one_input_node)
        _data = np.asarray(data, dtype=np.uint8)

        return OfflineFrameGen.gen_work_frame1_fast(common_frame_dest, _data)

    @staticmethod
    def gen_work_frame1_fast(
        frame_dest_info: FrameArrayType, data: NDArray[np.uint8]
    ) -> FrameArrayType:
        _max, _min = np.max(data, axis=None), np.min(data, axis=None)

        if _min < np.iinfo(np.uint8).min or _max > np.iinfo(np.uint8).max:
            raise ValueError(f"data out of range np.int8 ({_min}, {_max}).")

        if frame_dest_info.size != data.size:
            raise ValueError(
                f"the size of frame dest info & data are not equal, {frame_dest_info.size} != {data.size}."
            )

        return OfflineWorkFrame1._gen_frame_fast(frame_dest_info, data.flatten())

    @staticmethod
    def gen_work_frame2(chip_coord: Coord, n_sync: IntScalarType) -> OfflineWorkFrame2:
        return OfflineWorkFrame2(chip_coord, int(n_sync))

    @staticmethod
    def gen_work_frame3(chip_coord: Coord) -> OfflineWorkFrame3:
        return OfflineWorkFrame3(chip_coord)

    @staticmethod
    def gen_work_frame4(chip_coord: Coord) -> OfflineWorkFrame4:
        return OfflineWorkFrame4(chip_coord)
