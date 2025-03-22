import warnings
from typing import Any, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ..coordinate import ChipCoord, Coord, CoordLike
from ..coordinate import ReplicationId as RId
from ..coordinate import RIdLike, to_coord, to_rid
from ..hw_defs import HwParams
from ..ram_model import NeuronAttrsChecker, NeuronDestInfoChecker
from ..reg_model import CoreRegChecker
from ..reg_types import core_mode_check
from .base import Frame, FramePackage
from .frame_defs import FrameFormat as FF
from .frame_defs import FrameHeader as FH
from .frame_defs import OfflineCoreRegFormat as Off_CRegF
from .frame_defs import OfflineNeuronRAMFormat as Off_NRAMF
from .frame_defs import OfflineWorkFrame1Format as Off_WF1F
from .types import *
from .utils import (
    OUT_OF_RANGE_WARNING,
    ShapeError,
    TruncationWarning,
    _mask,
    bin_split,
    params_check,
    params_check2,
)

__all__ = [
    "OfflineConfigFrame1",
    "OfflineConfigFrame2",
    "OfflineConfigFrame3",
    "OfflineConfigFrame4",
    "OfflineTestInFrame1",
    "OfflineTestInFrame2",
    "OfflineTestInFrame3",
    "OfflineTestInFrame4",
    "OfflineTestOutFrame1",
    "OfflineTestOutFrame2",
    "OfflineTestOutFrame3",
    "OfflineTestOutFrame4",
    "OfflineWorkFrame1",
    "OfflineWorkFrame2",
    "OfflineWorkFrame3",
    "OfflineWorkFrame4",
]


_L_PACKAGE_TYPE_CONF_TESTOUT = 0b0  # Literal value of package type for conf & test-out
_L_PACKAGE_TYPE_TESTIN = 0b1  # Literal value of package type for test-in.


class _RandomSeedFrame(Frame):
    def __init__(
        self,
        header: FH,
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        random_seed: int,
    ) -> None:
        if random_seed > FF.GENERAL_MASK:
            warnings.warn(
                OUT_OF_RANGE_WARNING.format(
                    "random_seed", FF.FRAME_LENGTH, random_seed
                ),
                TruncationWarning,
            )

        payload = self._random_seed_split(random_seed)
        super().__init__(header, chip_coord, core_coord, rid, payload)

    @staticmethod
    def _random_seed_split(random_seed: int) -> FrameArrayType:
        _seed = random_seed & FF.GENERAL_MASK

        return np.asarray(
            [
                (_seed >> 34) & FF.GENERAL_PAYLOAD_MASK,
                (_seed >> 4) & FF.GENERAL_PAYLOAD_MASK,
                (_seed & _mask(4)) << (FF.GENERAL_PAYLOAD_LENGTH - 4),
            ],
            dtype=FRAME_DTYPE,
        )


class _CoreRegFrame(Frame):
    def __init__(
        self,
        header: FH,
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        params_reg_dict: dict[str, Any],
    ) -> None:
        payload = self._payload_reorganized(params_reg_dict)

        super().__init__(header, chip_coord, core_coord, rid, payload)

    @staticmethod
    @params_check(CoreRegChecker)
    @core_mode_check
    def _payload_reorganized(reg_dict: dict[str, Any]) -> FrameArrayType:
        # High 8 bits & low 7 bits of tick_wait_start
        tws_high8, tws_low7 = bin_split(reg_dict["tick_wait_start"], 7, 8)
        # High 3 bits & low 7 bits of test_chip_addr
        tca_high3, tca_low7 = bin_split(reg_dict["test_chip_addr"], 7, 3)

        reg_frame1 = (
            (reg_dict["weight_width"] & Off_CRegF.WEIGHT_WIDTH_MASK)
            << Off_CRegF.WEIGHT_WIDTH_OFFSET
            | ((reg_dict["LCN"] & Off_CRegF.LCN_MASK) << Off_CRegF.LCN_OFFSET)
            | (
                (reg_dict["input_width"] & Off_CRegF.INPUT_WIDTH_MASK)
                << Off_CRegF.INPUT_WIDTH_OFFSET
            )
            | (
                (reg_dict["spike_width"] & Off_CRegF.SPIKE_WIDTH_MASK)
                << Off_CRegF.SPIKE_WIDTH_OFFSET
            )
            | (
                (reg_dict["num_dendrite"] & Off_CRegF.NUM_VALID_DENDRITE_MASK)
                << Off_CRegF.NUM_VALID_DENDRITE_OFFSET
            )
            | (
                (reg_dict["pool_max"] & Off_CRegF.POOL_MAX_MASK)
                << Off_CRegF.POOL_MAX_OFFSET
            )
            | (
                (tws_high8 & Off_CRegF.TICK_WAIT_START_HIGH8_MASK)
                << Off_CRegF.TICK_WAIT_START_HIGH8_OFFSET
            )
        )

        reg_frame2 = (
            (
                (tws_low7 & Off_CRegF.TICK_WAIT_START_LOW7_MASK)
                << Off_CRegF.TICK_WAIT_START_LOW7_OFFSET
            )
            | (
                (reg_dict["tick_wait_end"] & Off_CRegF.TICK_WAIT_END_MASK)
                << Off_CRegF.TICK_WAIT_END_OFFSET
            )
            | ((reg_dict["snn_en"] & Off_CRegF.SNN_EN_MASK) << Off_CRegF.SNN_EN_OFFSET)
            | (
                (reg_dict["target_LCN"] & Off_CRegF.TARGET_LCN_MASK)
                << Off_CRegF.TARGET_LCN_OFFSET
            )
            | (
                (tca_high3 & Off_CRegF.TEST_CHIP_ADDR_HIGH3_MASK)
                << Off_CRegF.TEST_CHIP_ADDR_HIGH3_OFFSET
            )
        )

        reg_frame3 = (
            tca_low7 & Off_CRegF.TEST_CHIP_ADDR_LOW7_MASK
        ) << Off_CRegF.TEST_CHIP_ADDR_LOW7_OFFSET

        return np.asarray([reg_frame1, reg_frame2, reg_frame3], dtype=FRAME_DTYPE)

    @property
    def params_reg(self) -> Union[int, FRAME_DTYPE, FrameArrayType]:
        return self.payload


class _NeuronRAMFrame(FramePackage):
    N_FRAME_PER_NEURON_RAM = 4

    def __init__(
        self,
        header: FH,
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        sram_base_addr: int,
        n_neuron: int,
        neuron_attrs: dict[str, Any],
        neuron_dest_info: dict[str, Any],
        repeat: int,
    ) -> None:
        n_package = self.N_FRAME_PER_NEURON_RAM * n_neuron * repeat
        payload = _package_arg_check(
            sram_base_addr, n_package, _L_PACKAGE_TYPE_CONF_TESTOUT
        )
        packages = self._get_packages(neuron_attrs, neuron_dest_info, n_neuron, repeat)

        super().__init__(header, chip_coord, core_coord, rid, payload, packages)

    @staticmethod
    @params_check2(NeuronAttrsChecker, NeuronDestInfoChecker)
    def _get_packages(
        attrs: dict[str, Any], dest_info: dict[str, Any], n_neuron: int, repeat: int
    ) -> FrameArrayType:
        voltage: int = attrs.get("voltage", 0)

        def _gen_ram_frame1_and_2(leak_v: int) -> tuple[int, int]:
            leak_v_high2, leak_v_low28 = bin_split(leak_v, 28, 2)

            # Package #1, [63:0]
            ram_frame1 = (
                ((voltage & Off_NRAMF.VJT_PRE_MASK) << Off_NRAMF.VJT_PRE_OFFSET)
                | (
                    (attrs["bit_truncate"] & Off_NRAMF.BIT_TRUNCATE_MASK)
                    << Off_NRAMF.BIT_TRUNCATE_OFFSET
                )
                | (
                    (attrs["weight_det_stoch"] & Off_NRAMF.WEIGHT_DET_STOCH_MASK)
                    << Off_NRAMF.WEIGHT_DET_STOCH_OFFSET
                )
                | (
                    (leak_v_low28 & Off_NRAMF.LEAK_V_LOW28_MASK)
                    << Off_NRAMF.LEAK_V_LOW28_OFFSET
                )
            )

            # Package #2, [127:64]
            ram_frame2 = (
                (
                    (leak_v_high2 & Off_NRAMF.LEAK_V_HIGH2_MASK)
                    << Off_NRAMF.LEAK_V_HIGH2_OFFSET
                )
                | (
                    (attrs["leak_det_stoch"] & Off_NRAMF.LEAK_DET_STOCH_MASK)
                    << Off_NRAMF.LEAK_DET_STOCH_OFFSET
                )
                | (
                    (attrs["leak_reversal_flag"] & Off_NRAMF.LEAK_REVERSAL_FLAG_MASK)
                    << Off_NRAMF.LEAK_REVERSAL_FLAG_OFFSET
                )
                | (
                    (attrs["threshold_pos"] & Off_NRAMF.THRESHOLD_POS_MASK)
                    << Off_NRAMF.THRESHOLD_POS_OFFSET
                )
                | (
                    (attrs["threshold_neg"] & Off_NRAMF.THRESHOLD_NEG_MASK)
                    << Off_NRAMF.THRESHOLD_NEG_OFFSET
                )
                | (
                    (attrs["threshold_neg_mode"] & Off_NRAMF.THRESHOLD_NEG_MODE_MASK)
                    << Off_NRAMF.THRESHOLD_NEG_MODE_OFFSET
                )
                | (
                    (threshold_mask_ctrl_low1 & Off_NRAMF.THRESHOLD_MASK_CTRL_LOW1_MASK)
                    << Off_NRAMF.THRESHOLD_MASK_CTRL_LOW1_OFFSET
                )
            )

            return ram_frame1, ram_frame2

        def _gen_ram_frame3() -> int:
            # Package #3, [191:128]
            return (
                (
                    (
                        threshold_mask_ctrl_high4
                        & Off_NRAMF.THRESHOLD_MASK_CTRL_HIGH4_MASK
                    )
                    << Off_NRAMF.THRESHOLD_MASK_CTRL_HIGH4_OFFSET
                )
                | (
                    (attrs["leak_post"] & Off_NRAMF.LEAK_POST_MASK)
                    << Off_NRAMF.LEAK_POST_OFFSET
                )
                | (
                    (attrs["reset_v"] & Off_NRAMF.RESET_V_MASK)
                    << Off_NRAMF.RESET_V_OFFSET
                )
                | (
                    (attrs["reset_mode"] & Off_NRAMF.RESET_MODE_MASK)
                    << Off_NRAMF.RESET_MODE_OFFSET
                )
                | (
                    (dest_info["addr_chip_y"] & Off_NRAMF.ADDR_CHIP_Y_MASK)
                    << Off_NRAMF.ADDR_CHIP_Y_OFFSET
                )
                | (
                    (dest_info["addr_chip_x"] & Off_NRAMF.ADDR_CHIP_X_MASK)
                    << Off_NRAMF.ADDR_CHIP_X_OFFSET
                )
                | (
                    (dest_info["addr_core_y_ex"] & Off_NRAMF.ADDR_CORE_Y_EX_MASK)
                    << Off_NRAMF.ADDR_CORE_Y_EX_OFFSET
                )
                | (
                    (dest_info["addr_core_x_ex"] & Off_NRAMF.ADDR_CORE_X_EX_MASK)
                    << Off_NRAMF.ADDR_CORE_X_EX_OFFSET
                )
                | (
                    (dest_info["addr_core_y"] & Off_NRAMF.ADDR_CORE_Y_MASK)
                    << Off_NRAMF.ADDR_CORE_Y_OFFSET
                )
                | (
                    (addr_core_x_low2 & Off_NRAMF.ADDR_CORE_X_LOW2_MASK)
                    << Off_NRAMF.ADDR_CORE_X_LOW2_OFFSET
                )
            )

        def _gen_ram_frame4(idx: int) -> int:
            # Package #4, [213:192]
            return (
                (
                    (addr_core_x_high3 & Off_NRAMF.ADDR_CORE_X_HIGH3_MASK)
                    << Off_NRAMF.ADDR_CORE_X_HIGH3_OFFSET
                )
                | (
                    (addr_axon[idx] & Off_NRAMF.ADDR_AXON_MASK)
                    << Off_NRAMF.ADDR_AXON_OFFSET
                )
                | (
                    (tick_relative[idx] & Off_NRAMF.TICK_RELATIVE_MASK)
                    << Off_NRAMF.TICK_RELATIVE_OFFSET
                )
            )

        tick_relative = dest_info["tick_relative"]
        addr_axon = dest_info["addr_axon"]

        if len(tick_relative) != len(addr_axon):
            raise ValueError(
                f"length of 'tick_relative' & 'addr_axon' are not equal, "
                f"{len(tick_relative)} != {len(addr_axon)}."
            )

        if n_neuron > len(tick_relative):
            raise ValueError(f"length of 'tick_relative' out of range {n_neuron}.")

        _packages = np.zeros((n_neuron, 4), dtype=FRAME_DTYPE)

        threshold_mask_ctrl_high4, threshold_mask_ctrl_low1 = bin_split(
            attrs["threshold_mask_ctrl"], 1, 4
        )
        addr_core_x_high3, addr_core_x_low2 = bin_split(dest_info["addr_core_x"], 2, 3)

        # LSB: [63:0], [127:64], [191:128], [213:192]
        ram_frame3 = _gen_ram_frame3()

        leak_v: Union[int, NDArray[np.int32]] = attrs["leak_v"]

        if isinstance(leak_v, int):
            ram_frame1, ram_frame2 = _gen_ram_frame1_and_2(leak_v)

            _package_common = np.array(
                [ram_frame1, ram_frame2, ram_frame3], dtype=FRAME_DTYPE
            )
            # Repeat the common part of the package
            _packages[:, :3] = np.tile(_package_common, (n_neuron, 1))

            # Iterate destination infomation of every neuron
            for i in range(n_neuron):
                ram_frame4 = _gen_ram_frame4(i)
                _packages[i][-1] = ram_frame4

        else:
            if leak_v.size != n_neuron:
                raise ValueError(
                    f"length of 'leak_v' is not equal to #N of neuron, "
                    f"{leak_v.size} != {n_neuron}."
                )

            for i in range(n_neuron):
                ram_frame1, ram_frame2 = _gen_ram_frame1_and_2(int(leak_v[i]))
                ram_frame4 = _gen_ram_frame4(i)
                _packages[i] = np.array(
                    [ram_frame1, ram_frame2, ram_frame3, ram_frame4], dtype=FRAME_DTYPE
                )

        # Tile the package of every neuron `repeat` times & flatten
        # (n_neuron, 4) -> (n_neuron * 4 * repeat,)
        packages_tiled = np.tile(_packages, repeat).ravel()

        return packages_tiled


class _WeightRAMFrame(FramePackage):
    def __init__(
        self,
        header: FH,
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        /,
        sram_base_addr: int,
        n_package: int,
        weight_ram: FrameArrayType,
    ) -> None:
        payload = _package_arg_check(
            sram_base_addr, n_package, _L_PACKAGE_TYPE_CONF_TESTOUT
        )
        _weight_ram = weight_ram.ravel()

        super().__init__(header, chip_coord, core_coord, rid, payload, _weight_ram)


class OfflineConfigFrame1(_RandomSeedFrame):
    header: FH = FH.CONFIG_TYPE1

    def __init__(
        self, test_chip_coord: Coord, core_coord: Coord, rid: RId, /, random_seed: int
    ) -> None:
        super().__init__(self.header, test_chip_coord, core_coord, rid, random_seed)


class OfflineConfigFrame2(_CoreRegFrame):
    header: FH = FH.CONFIG_TYPE2

    def __init__(
        self,
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        /,
        params_reg_dict: dict[str, Any],
    ) -> None:
        super().__init__(self.header, chip_coord, core_coord, rid, params_reg_dict)


class OfflineConfigFrame3(_NeuronRAMFrame):
    header: FH = FH.CONFIG_TYPE3

    def __init__(
        self,
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        /,
        sram_base_addr: int,
        n_neuron: int,
        neuron_attrs: dict[str, Any],
        neuron_dest_info: dict[str, Any],
        repeat: int,
    ) -> None:
        super().__init__(
            self.header,
            chip_coord,
            core_coord,
            rid,
            sram_base_addr,
            n_neuron,
            neuron_attrs,
            neuron_dest_info,
            repeat,
        )


class OfflineConfigFrame4(_WeightRAMFrame):
    header: FH = FH.CONFIG_TYPE4

    def __init__(
        self,
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        /,
        sram_base_addr: int,
        n_package: int,
        weight_ram: FrameArrayType,
    ) -> None:
        super().__init__(
            self.header,
            chip_coord,
            core_coord,
            rid,
            sram_base_addr,
            n_package,
            weight_ram,
        )


class OfflineTestInFrame1(Frame):
    header: FH = FH.TEST_TYPE1

    def __init__(self, chip_coord: ChipCoord, core_coord: Coord, rid: RId, /) -> None:
        super().__init__(self.header, chip_coord, core_coord, rid, FRAME_DTYPE(0))


class OfflineTestOutFrame1(_RandomSeedFrame):
    header: FH = FH.TEST_TYPE1

    def __init__(
        self,
        test_chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        random_seed: int,
    ) -> None:
        super().__init__(self.header, test_chip_coord, core_coord, rid, random_seed)


class OfflineTestInFrame2(Frame):
    header: FH = FH.TEST_TYPE2

    def __init__(self, chip_coord: ChipCoord, core_coord: Coord, rid: RId, /) -> None:
        super().__init__(self.header, chip_coord, core_coord, rid, FRAME_DTYPE(0))


class OfflineTestOutFrame2(_CoreRegFrame):
    header: FH = FH.TEST_TYPE2

    def __init__(
        self,
        test_chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        params_reg_dict: dict[str, Any],
    ) -> None:
        super().__init__(self.header, test_chip_coord, core_coord, rid, params_reg_dict)


class OfflineTestInFrame3(Frame):
    header: FH = FH.TEST_TYPE3

    def __init__(
        self,
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        /,
        sram_base_addr: int,
        n_package: int,
    ) -> None:
        payload = _package_arg_check(sram_base_addr, n_package, _L_PACKAGE_TYPE_TESTIN)
        super().__init__(self.header, chip_coord, core_coord, rid, payload)


class OfflineTestOutFrame3(_NeuronRAMFrame):
    header: FH = FH.TEST_TYPE3

    def __init__(
        self,
        test_chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        sram_base_addr: int,
        n_neuron: int,
        neuron_attrs: dict[str, Any],
        neuron_dest_info: dict[str, Any],
        repeat: int = 1,
    ) -> None:
        super().__init__(
            self.header,
            test_chip_coord,
            core_coord,
            rid,
            sram_base_addr,
            n_neuron,
            neuron_attrs,
            neuron_dest_info,
            repeat,
        )


class OfflineTestInFrame4(Frame):
    header: FH = FH.TEST_TYPE4

    def __init__(
        self,
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        /,
        sram_base_addr: int,
        n_package: int,
    ):
        payload = _package_arg_check(sram_base_addr, n_package, _L_PACKAGE_TYPE_TESTIN)
        super().__init__(self.header, chip_coord, core_coord, rid, payload)


class OfflineTestOutFrame4(_WeightRAMFrame):
    header: FH = FH.TEST_TYPE4

    def __init__(
        self,
        test_chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        sram_base_addr: int,
        n_package: int,
        weight_ram: FrameArrayType,
    ) -> None:
        super().__init__(
            self.header,
            test_chip_coord,
            core_coord,
            rid,
            sram_base_addr,
            n_package,
            weight_ram,
        )


class OfflineWorkFrame1(Frame):
    header: FH = FH.WORK_TYPE1

    def __init__(
        self,
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        /,
        timeslot: IntScalarType,
        axon: IntScalarType,
        data: DataType,  # signed int8
    ) -> None:
        self._axon = int(axon)
        self._timeslot = int(timeslot)

        if self._timeslot > Off_WF1F.TIMESLOT_MASK or self._timeslot < 0:
            raise ValueError(
                f"timeslot out of range {Off_WF1F.TIMESLOT_MASK} ({self._timeslot})."
            )

        if self._axon > HwParams.ADDR_AXON_MAX or self._axon < 0:
            raise ValueError(
                f"axon out of range {HwParams.ADDR_AXON_MAX} ({self._axon})."
            )

        if isinstance(data, np.ndarray) and data.size != 1:
            raise ShapeError(f"size of data must be 1 ({data.size}).")

        if data < np.iinfo(np.int8).min or data > np.iinfo(np.int8).max:
            raise ValueError(f"data out of range np.int8 ({data}).")

        self.data = PAYLOAD_DATA_DTYPE(data)

        payload = FRAME_DTYPE(
            ((self._axon & Off_WF1F.AXON_MASK) << Off_WF1F.AXON_OFFSET)
            | ((self._timeslot & Off_WF1F.TIMESLOT_MASK) << Off_WF1F.TIMESLOT_OFFSET)
            | ((data & Off_WF1F.DATA_MASK) << Off_WF1F.DATA_OFFSET)
        )

        super().__init__(self.header, chip_coord, core_coord, rid, payload)

    @property
    def target_timeslot(self) -> int:
        return self._timeslot

    @property
    def target_axon(self) -> int:
        return self._axon

    @staticmethod
    @params_check(NeuronDestInfoChecker)
    def _frame_dest_reorganized(dest_info: dict[str, Any]) -> FrameArrayType:
        return OfflineWorkFrame1.concat_frame_dest(
            (dest_info["addr_chip_x"], dest_info["addr_chip_y"]),
            (dest_info["addr_core_x"], dest_info["addr_core_y"]),
            (dest_info["addr_core_x_ex"], dest_info["addr_core_y_ex"]),
            dest_info["addr_axon"],
            dest_info["tick_relative"],
        )

    @staticmethod
    def _gen_frame_fast(
        frame_dest_info: FrameArrayType, data: PayloadDataType
    ) -> FrameArrayType:
        """DO NOT call `OfflineWorkFrame1._gen_frame_fast()` directly."""
        indexes = np.nonzero(data)

        return frame_dest_info[indexes] + data[indexes]

    @classmethod
    def concat_frame_dest(
        cls,
        chip_coord: CoordLike,
        core_coord: CoordLike,
        rid: RIdLike,
        /,
        axons: ArrayType,
        timeslots: Optional[ArrayType] = None,
    ) -> FrameArrayType:
        _axons = np.asarray(axons, dtype=FRAME_DTYPE).ravel()

        if timeslots is not None:
            _timeslots = np.asarray(timeslots, dtype=FRAME_DTYPE).ravel()
        else:
            _timeslots = np.zeros_like(_axons)

        if _axons.size != _timeslots.size:
            raise ValueError(
                f"the size of axons & timeslots are not equal, {_axons.size} !={_timeslots.size}."
            )

        _chip_coord = to_coord(chip_coord)
        _core_coord = to_coord(core_coord)
        _rid = to_rid(rid)

        header = cls.header.value & FF.GENERAL_HEADER_MASK
        chip_addr = _chip_coord.address & FF.GENERAL_CHIP_ADDR_MASK
        core_addr = _core_coord.address & FF.GENERAL_CORE_ADDR_MASK
        rid_addr = _rid.address & FF.GENERAL_CORE_EX_ADDR_MASK

        common_head = (
            (header << FF.GENERAL_HEADER_OFFSET)
            + (chip_addr << FF.GENERAL_CHIP_ADDR_OFFSET)
            + (core_addr << FF.GENERAL_CORE_ADDR_OFFSET)
            + (rid_addr << FF.GENERAL_CORE_EX_ADDR_OFFSET)
        )

        common_payload = ((_axons & Off_WF1F.AXON_MASK) << Off_WF1F.AXON_OFFSET) | (
            (_timeslots & Off_WF1F.TIMESLOT_MASK) << Off_WF1F.TIMESLOT_OFFSET
        )

        return (common_head + common_payload).astype(FRAME_DTYPE)


class OfflineWorkFrame2(Frame):
    header: FH = FH.WORK_TYPE2

    def __init__(self, chip_coord: ChipCoord, /, n_sync: int) -> None:
        if n_sync > FF.GENERAL_PAYLOAD_MASK:
            warnings.warn(
                OUT_OF_RANGE_WARNING.format("n_sync", 30, n_sync), TruncationWarning
            )

        super().__init__(
            self.header,
            chip_coord,
            Coord(0, 0),
            RId(0, 0),
            FRAME_DTYPE(n_sync & FF.GENERAL_PAYLOAD_MASK),
        )


class OfflineWorkFrame3(Frame):
    header: FH = FH.WORK_TYPE3

    def __init__(self, chip_coord: ChipCoord) -> None:
        super().__init__(
            self.header, chip_coord, Coord(0, 0), RId(0, 0), FRAME_DTYPE(0)
        )


class OfflineWorkFrame4(Frame):
    header: FH = FH.WORK_TYPE4

    def __init__(self, chip_coord: ChipCoord) -> None:
        super().__init__(
            self.header, chip_coord, Coord(0, 0), RId(0, 0), FRAME_DTYPE(0)
        )


def _package_arg_check(
    sram_base_addr: int, n_package: int, package_type: Literal[0, 1]
) -> FRAME_DTYPE:
    if sram_base_addr > Off_NRAMF.GENERAL_PACKAGE_SRAM_ADDR_MASK or sram_base_addr < 0:
        raise ValueError(f"SRAM base address out of range, {sram_base_addr}.")

    if n_package > Off_NRAMF.GENERAL_PACKAGE_NUM_MASK or n_package < 0:
        raise ValueError(f"the numeber of data package out of range, {n_package}.")

    return FRAME_DTYPE(
        (
            (sram_base_addr & FF.GENERAL_PACKAGE_SRAM_ADDR_MASK)
            << FF.GENERAL_PACKAGE_SRAM_ADDR_OFFSET
        )
        | (
            (package_type & FF.GENERAL_PACKAGE_TYPE_MASK)
            << FF.GENERAL_PACKAGE_TYPE_OFFSET
        )
        | ((n_package & FF.GENERAL_PACKAGE_NUM_MASK) << FF.GENERAL_PACKAGE_NUM_OFFSET)
    )
