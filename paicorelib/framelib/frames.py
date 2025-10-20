import warnings
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..coordinate import ChipCoord, Coord, CoordLike
from ..coordinate import ReplicationId as RId
from ..coordinate import RIdLike, to_coord, to_rid
from ..hw_defs import HwOfflineCoreParams as OffCoreParams
from ..hw_defs import HwOnlineCoreParams as OnCoreParams
from ..ram_model import NeuAttrs, NeuDestInfo
from ..ram_model import OfflineNeuAttrs as OffNeuAttrs
from ..ram_model import OfflineNeuDestInfo as OffNeuDestInfo
from ..ram_model import OfflineNeuDestInfoChecker as OffNeuDestInfoChecker
from ..ram_model import OnlineNeuAttrs as OnNeuAttrs
from ..ram_model import OnlineNeuDestInfo as OnNeuDestInfo
from ..ram_model import OnlineNeuDestInfoChecker as OnNeuDestInfoChecker
from ..reg_defs import WeightWidth, core_mode_check
from ..reg_model import CoreReg
from ..reg_model import OfflineCoreReg as OffCoreReg
from ..reg_model import OnlineCoreReg as OnCoreReg
from ..routing_defs import _rid_unset
from .base import Frame, FramePackage, FramePackagePayload, _get_frame_common
from .frame_defs import FrameFormat as FF
from .frame_defs import FrameHeader as FH
from .frame_defs import FramePackageType as FPType
from .frame_defs import OfflineCoreRegFormat as Off_CRegF
from .frame_defs import OfflineNeuRAMFormat as Off_NRAMF
from .frame_defs import OfflineWorkFrame1Format as Off_WF1F
from .frame_defs import OfflineWorkFrame2Format as Off_WF2F
from .frame_defs import Online_WF1F_SubType
from .frame_defs import OnlineCoreRegFormat as On_CRegF
from .frame_defs import OnlineNeuRAMFormat as On_NRAMF
from .frame_defs import OnlineNeuRAMFormat_WW1 as On_NRAMF_WW1
from .frame_defs import OnlineNeuRAMFormat_WWn as On_NRAMF_WWn
from .frame_defs import OnlineWorkFrame1Format as On_WF1F
from .frame_defs import OnlineWorkFrame1Format_1 as On_WF1F_1
from .types import *
from .utils import (
    OUT_OF_RANGE_WARNING,
    ShapeError,
    TruncationWarning,
    _mask,
    bin_split,
    params_check,
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
    "OnlineConfigFrame1",
    "OnlineConfigFrame2",
    "OnlineConfigFrame3",
    "OnlineConfigFrame4",
    "OnlineTestInFrame1",
    "OnlineTestInFrame2",
    "OnlineTestInFrame3",
    "OnlineTestInFrame4",
    "OnlineTestOutFrame1",
    "OnlineTestOutFrame2",
    "OnlineTestOutFrame3",
    "OnlineTestOutFrame4",
    "OnlineWorkFrame1_1",
    "OnlineWorkFrame1_2",
    "OnlineWorkFrame1_3",
    "OnlineWorkFrame1_4",
    "OnlineWorkFrame2",
    "OnlineWorkFrame3",
    "OnlineWorkFrame4",
]


class _RandomSeedFrame(Frame):
    N_FRAME_PAYLOAD: ClassVar[int] = 3

    def __init__(
        self, chip_coord: ChipCoord, core_coord: Coord, rid: RId, random_seed: int
    ) -> None:
        if random_seed > FF.GENERAL_MASK:
            warnings.warn(
                OUT_OF_RANGE_WARNING.format(
                    "random_seed", FF.FRAME_LENGTH, random_seed
                ),
                TruncationWarning,
            )

        payload = self._random_seed_split(random_seed)
        super().__init__(chip_coord, core_coord, rid, payload)

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
    N_FRAME_PAYLOAD: ClassVar[int]

    def __init__(
        self, chip_coord: ChipCoord, core_coord: Coord, rid: RId, core_reg: CoreReg
    ) -> None:
        core_reg_dict = core_reg.model_dump()
        payload = self.payload_reorganized(core_reg_dict)
        super().__init__(chip_coord, core_coord, rid, payload)

    @staticmethod
    @abstractmethod
    def payload_reorganized(core_reg: dict[str, Any]) -> FrameArrayType: ...

    @property
    def params_reg(self) -> FRAME_DTYPE | FrameArrayType:
        return self.payload


@overload
def _convert_core_reg(
    core_reg: OffCoreReg | dict[str, Any], type: Literal["offline"]
) -> OffCoreReg: ...


@overload
def _convert_core_reg(
    core_reg: OnCoreReg | dict[str, Any], type: Literal["online"]
) -> OnCoreReg: ...


def _convert_core_reg(
    core_reg: CoreReg | dict[str, Any], type: Literal["offline", "online"]
) -> OffCoreReg | OnCoreReg:
    if isinstance(core_reg, (OffCoreReg, OnCoreReg)):
        return core_reg
    else:
        if type == "offline":
            return OffCoreReg.model_validate(core_reg, strict=True)
        else:
            return OnCoreReg.model_validate(core_reg, strict=True)


class _OfflineCoreRegFrame(_CoreRegFrame):
    N_FRAME_PAYLOAD = 3

    def __init__(
        self,
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        core_reg: OffCoreReg | dict[str, Any],
    ) -> None:
        _core_reg = _convert_core_reg(core_reg, "offline")
        super().__init__(chip_coord, core_coord, rid, _core_reg)

    @staticmethod
    @core_mode_check
    def payload_reorganized(core_reg: dict[str, Any]) -> FrameArrayType:
        # High 8 bits & low 7 bits of 'tick_wait_start'
        tws_high8, tws_low7 = bin_split(core_reg["tick_wait_start"], 7, 8)
        reg_frame1 = (
            (core_reg["weight_width"] & Off_CRegF.WEIGHT_WIDTH_MASK)
            << Off_CRegF.WEIGHT_WIDTH_OFFSET
            | ((core_reg["lcn"] & Off_CRegF.LCN_MASK) << Off_CRegF.LCN_OFFSET)
            | (
                (core_reg["input_width"] & Off_CRegF.INPUT_WIDTH_MASK)
                << Off_CRegF.INPUT_WIDTH_OFFSET
            )
            | (
                (core_reg["spike_width"] & Off_CRegF.SPIKE_WIDTH_MASK)
                << Off_CRegF.SPIKE_WIDTH_OFFSET
            )
            | (
                (core_reg["num_dendrite"] & Off_CRegF.NUM_DENDRITE_MASK)
                << Off_CRegF.NUM_DENDRITE_OFFSET
            )
            | (
                (core_reg["max_pooling_en"] & Off_CRegF.MAX_POOLING_EN_MASK)
                << Off_CRegF.MAX_POOLING_EN_OFFSET
            )
            | (
                (tws_high8 & Off_CRegF.TICK_WAIT_START_HIGH8_MASK)
                << Off_CRegF.TICK_WAIT_START_HIGH8_OFFSET
            )
        )

        # High 3 bits & low 7 bits of 'test_chip_addr'
        tca_high3, tca_low7 = bin_split(core_reg["test_chip_addr"], 7, 3)
        reg_frame2 = (
            (
                (tws_low7 & Off_CRegF.TICK_WAIT_START_LOW7_MASK)
                << Off_CRegF.TICK_WAIT_START_LOW7_OFFSET
            )
            | (
                (core_reg["tick_wait_end"] & Off_CRegF.TICK_WAIT_END_MASK)
                << Off_CRegF.TICK_WAIT_END_OFFSET
            )
            | ((core_reg["snn_en"] & Off_CRegF.SNN_EN_MASK) << Off_CRegF.SNN_EN_OFFSET)
            | (
                (core_reg["target_lcn"] & Off_CRegF.TARGET_LCN_MASK)
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


class _OnlineCoreRegFrame(_CoreRegFrame):
    N_FRAME_PAYLOAD = 8

    def __init__(
        self,
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        core_reg: OnCoreReg | dict[str, Any],
    ) -> None:
        _core_reg = _convert_core_reg(core_reg, "online")
        super().__init__(chip_coord, core_coord, rid, _core_reg)

    @staticmethod
    def payload_reorganized(core_reg: dict[str, Any]) -> FrameArrayType:
        # High 26 bits & low 6 bits of 'lateral_inhi_value'
        lateral_inhi_high26, lateral_inhi_low6 = bin_split(
            core_reg["lateral_inhi_value"], 6, 26
        )
        reg_frame1 = (
            (core_reg["weight_width"] & On_CRegF.WEIGHT_WIDTH_MASK)
            << On_CRegF.WEIGHT_WIDTH_OFFSET
            | ((core_reg["lcn"] & On_CRegF.LCN_MASK) << On_CRegF.LCN_OFFSET)
            | (
                (lateral_inhi_high26 & On_CRegF.LATERAL_INHI_VALUE_HIGH26_MASK)
                << On_CRegF.LATERAL_INHI_VALUE_HIGH26_OFFSET
            )
        )

        reg_frame2 = (
            (lateral_inhi_low6 & On_CRegF.LATERAL_INHI_VALUE_LOW6_MASK)
            << On_CRegF.LATERAL_INHI_VALUE_LOW6_OFFSET
            | (
                (core_reg["weight_decay_value"] & On_CRegF.WEIGHT_DECAY_MASK)
                << On_CRegF.WEIGHT_DECAY_OFFSET
            )
            | (
                (core_reg["upper_weight"] & On_CRegF.UPPER_WEIGHT_MASK)
                << On_CRegF.UPPER_WEIGHT_OFFSET
            )
            | (
                (core_reg["lower_weight"] & On_CRegF.LOWER_WEIGHT_MASK)
                << On_CRegF.LOWER_WEIGHT_OFFSET
            )
        )

        reg_frame3 = (
            (
                (core_reg["neuron_start"] & On_CRegF.NEURON_START_MASK)
                << On_CRegF.NEURON_START_OFFSET
            )
            | (
                (core_reg["neuron_end"] & On_CRegF.NEURON_END_MASK)
                << On_CRegF.NEURON_END_OFFSET
            )
            | (
                (core_reg["inhi_core_x_ex"] & On_CRegF.INHI_CORE_X_EX_MASK)
                << On_CRegF.INHI_CORE_X_EX_OFFSET
            )
            | (
                (core_reg["inhi_core_y_ex"] & On_CRegF.INHI_CORE_Y_EX_MASK)
                << On_CRegF.INHI_CORE_Y_EX_OFFSET
            )
        )

        reg_frame4 = (
            (core_reg["tick_wait_start"] & On_CRegF.TICK_WAIT_START_MASK)
            << On_CRegF.TICK_WAIT_START_OFFSET
        ) | (
            (core_reg["tick_wait_end"] & On_CRegF.TICK_WAIT_END_MASK)
            << On_CRegF.TICK_WAIT_END_OFFSET
        )

        # High 30 bits & low 30 bits of 'lut_random_en'
        lut_random_en_high30, lut_random_en_low30 = bin_split(
            core_reg["lut_random_en"], 30, 30
        )
        reg_frame5 = (
            lut_random_en_high30 & On_CRegF.LUT_RANDOM_EN_HIGH30_MASK
        ) << On_CRegF.LUT_RANDOM_EN_HIGH30_OFFSET
        reg_frame6 = (
            lut_random_en_low30 & On_CRegF.LUT_RANDOM_EN_LOW30_MASK
        ) << On_CRegF.LUT_RANDOM_EN_LOW30_OFFSET

        reg_frame7 = (
            (
                (core_reg["decay_random_en"] & On_CRegF.DECAY_RANDOM_EN_MASK)
                << On_CRegF.DECAY_RANDOM_EN_MASK
            )
            | (
                (core_reg["leak_order"] & On_CRegF.LEAK_ORDER_MASK)
                << On_CRegF.LEAK_ORDER_OFFSET
            )
            | (
                (core_reg["online_mode_en"] & On_CRegF.ONLINE_MODE_EN_MASK)
                << On_CRegF.ONLINE_MODE_EN_OFFSET
            )
            | (
                (core_reg["test_chip_addr"] & On_CRegF.TEST_CHIP_ADDR_MASK)
                << On_CRegF.TEST_CHIP_ADDR_OFFSET
            )
            | (
                (core_reg["random_seed"] & On_CRegF.RANDOM_SEED_MASK)
                << On_CRegF.RANDOM_SEED_OFFSET
            )
        )
        reg_frame8 = 0  # 30'd0

        return np.asarray(
            [
                reg_frame1,
                reg_frame2,
                reg_frame3,
                reg_frame4,
                reg_frame5,
                reg_frame6,
                reg_frame7,
                reg_frame8,
            ],
            dtype=FRAME_DTYPE,
        )


class _NeuRAMFrame(FramePackage, ABC):
    N_FRAME_PAYLOAD: int

    def __init__(
        self,
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        neu_start_addr: int,
        n_neuron: int,
        neu_attrs: NeuAttrs,
        neu_dest_info: NeuDestInfo,
        repeat: int,
    ) -> None:
        n_package = self.N_FRAME_PAYLOAD * n_neuron * repeat
        payload = FramePackagePayload(neu_start_addr, FPType.CONF_TESTOUT, n_package)
        neu_attrs_dict = neu_attrs.model_dump()
        neu_dest_info_dict = neu_dest_info.model_dump()
        packages = self.get_packages(
            neu_attrs_dict, neu_dest_info_dict, n_neuron, repeat
        )

        super().__init__(chip_coord, core_coord, rid, payload, packages)

    @abstractmethod
    def get_packages(
        self,
        attrs: dict[str, Any],
        dest_info: dict[str, Any],
        n_neuron: int,
        repeat: int,
    ) -> FrameArrayType: ...


@overload
def _convert_neu_attrs_dest_info(
    neu_attrs: OffNeuAttrs | dict[str, Any],
    neu_dest_info: OffNeuDestInfo | dict[str, Any],
    type: Literal["offline"],
) -> tuple[OffNeuAttrs, OffNeuDestInfo]: ...


@overload
def _convert_neu_attrs_dest_info(
    neu_attrs: OnNeuAttrs | dict[str, Any],
    neu_dest_info: OnNeuDestInfo | dict[str, Any],
    type: Literal["online"],
    ww: WeightWidth,
) -> tuple[OnNeuAttrs, OnNeuDestInfo]: ...


def _convert_neu_attrs_dest_info(
    neu_attrs: NeuAttrs | dict[str, Any],
    neu_dest_info: NeuDestInfo | dict[str, Any],
    type: Literal["offline", "online"],
    ww: WeightWidth | None = None,
):
    if isinstance(neu_attrs, (OffNeuAttrs, OnNeuAttrs)):
        _neu_attrs = neu_attrs
    else:
        if type == "offline":
            _neu_attrs = OffNeuAttrs.model_validate(neu_attrs, strict=True)
        else:
            _neu_attrs = OnNeuAttrs.model_validate(
                neu_attrs, context={"weight_width": ww}, strict=True
            )

    if isinstance(neu_dest_info, (OffNeuDestInfo, OnNeuDestInfo)):
        _neu_dest_info = neu_dest_info
    else:
        if type == "offline":
            _neu_dest_info = OffNeuDestInfo.model_validate(neu_dest_info, strict=True)
        else:
            _neu_dest_info = OnNeuDestInfo.model_validate(neu_dest_info, strict=True)

    return _neu_attrs, _neu_dest_info


class _OfflineNeuRAMFrame(_NeuRAMFrame):
    N_FRAME_PAYLOAD = 4  # One neuron needs 4 frames to configure

    def __init__(
        self,
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        neu_start_addr: int,
        n_neuron: int,
        neu_attrs: OffNeuAttrs | dict[str, Any],
        neu_dest_info: OffNeuDestInfo | dict[str, Any],
        repeat: int,
    ) -> None:
        _neu_attrs, _neu_dest_info = _convert_neu_attrs_dest_info(
            neu_attrs, neu_dest_info, "offline"
        )

        super().__init__(
            chip_coord,
            core_coord,
            rid,
            neu_start_addr,
            n_neuron,
            _neu_attrs,
            _neu_dest_info,
            repeat,
        )

    def get_packages(
        self,
        attrs: dict[str, Any],
        dest_info: dict[str, Any],
        n_neuron: int,
        repeat: int,
    ) -> FrameArrayType:
        voltage: int = attrs.get("voltage", 0)

        def _gen_ram_frame1_and_2(leak_v: int) -> tuple[FRAME_DTYPE, FRAME_DTYPE]:
            leak_v_high2, leak_v_low28 = bin_split(leak_v, 28, 2)

            # Package #1, [63:0]
            rf1 = (
                ((voltage & Off_NRAMF.VOLTAGE_MASK) << Off_NRAMF.VOLTAGE_OFFSET)
                | (
                    (attrs["bit_trunc"] & Off_NRAMF.BIT_TRUNC_MASK)
                    << Off_NRAMF.BIT_TRUNC_OFFSET
                )
                | (
                    (
                        attrs["syn_integration_mode"]
                        & Off_NRAMF.SYN_INTEGRATION_MODE_MASK
                    )
                    << Off_NRAMF.SYN_INTEGRATION_MODE_OFFSET
                )
                | (
                    (leak_v_low28 & Off_NRAMF.LEAK_V_LOW28_MASK)
                    << Off_NRAMF.LEAK_V_LOW28_OFFSET
                )
            )
            # Package #2, [127:64]
            rf2 = (
                (
                    (leak_v_high2 & Off_NRAMF.LEAK_V_HIGH2_MASK)
                    << Off_NRAMF.LEAK_V_HIGH2_OFFSET
                )
                | (
                    (
                        attrs["leak_integration_mode"]
                        & Off_NRAMF.LEAK_INTEGRATION_MODE_MASK
                    )
                    << Off_NRAMF.LEAK_INTEGRATION_MODE_OFFSET
                )
                | (
                    (attrs["leak_direction"] & Off_NRAMF.LEAK_DIRECTION_MASK)
                    << Off_NRAMF.LEAK_DIRECTION_OFFSET
                )
                | (
                    (attrs["pos_threshold"] & Off_NRAMF.POS_THRESHOLD_MASK)
                    << Off_NRAMF.POS_THRESHOLD_OFFSET
                )
                | (
                    (attrs["neg_threshold"] & Off_NRAMF.NEG_THRES_MASK)
                    << Off_NRAMF.NEG_THRES_OFFSET
                )
                | (
                    (attrs["neg_thres_mode"] & Off_NRAMF.NEG_THRES_MODE_MASK)
                    << Off_NRAMF.NEG_THRES_MODE_OFFSET
                )
                | (
                    (thres_mask_bit_low1 & Off_NRAMF.THRE_MASK_BITS_LOW1_MASK)
                    << Off_NRAMF.THRE_MASK_BITS_LOW1_OFFSET
                )
            )

            return FRAME_DTYPE(rf1), FRAME_DTYPE(rf2)

        def _gen_ram_frame3() -> FRAME_DTYPE:
            # Package #3, [191:128]
            return FRAME_DTYPE(
                (
                    (thres_mask_bit_high4 & Off_NRAMF.THRES_MASK_BITS_HIGH4_MASK)
                    << Off_NRAMF.THRES_MASK_BITS_HIGH4_OFFSET
                )
                | (
                    (attrs["leak_comparison"] & Off_NRAMF.LEAK_COMPARISON_MASK)
                    << Off_NRAMF.LEAK_COMPARISON_OFFSET
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

        def _gen_ram_frame4(idx: int) -> FRAME_DTYPE:
            # Package #4, [213:192]
            return FRAME_DTYPE(
                (
                    (addr_core_x_high3 & Off_NRAMF.ADDR_CORE_X_HIGH3_MASK)
                    << Off_NRAMF.ADDR_CORE_X_HIGH3_OFFSET
                )
                | (
                    (addr_axon[idx] & Off_NRAMF.ADDR_AXON_MASK)
                    << Off_NRAMF.ADDR_AXON_OFFSET
                )
                | (
                    (timeslot[idx] & Off_NRAMF.TICK_RELATIVE_MASK)
                    << Off_NRAMF.TICK_RELATIVE_OFFSET
                )
            )

        timeslot = dest_info["tick_relative"]
        addr_axon = dest_info["addr_axon"]

        if len(timeslot) != len(addr_axon):
            raise ValueError(
                f"length of 'tick_relative' & 'addr_axon' are not equal, "
                f"{len(timeslot)} != {len(addr_axon)}"
            )

        if n_neuron > len(timeslot):
            raise ValueError(f"length of 'tick_relative' out of range ({n_neuron})")

        packages = np.zeros((n_neuron, 4), dtype=FRAME_DTYPE)
        thres_mask_bit_high4, thres_mask_bit_low1 = bin_split(
            attrs["thres_mask_bits"], 1, 4
        )
        addr_core_x_high3, addr_core_x_low2 = bin_split(dest_info["addr_core_x"], 2, 3)

        # LSB: [63:0], [127:64], [191:128], [213:192]
        rf3 = _gen_ram_frame3()

        leak_v: int | NDArray[np.int32] = attrs["leak_v"]
        if isinstance(leak_v, np.ndarray):
            if leak_v.size != n_neuron:
                raise ValueError(
                    f"length of 'leak_v' is not equal to #N of neuron, "
                    f"{leak_v.size} != {n_neuron}"
                )
        else:
            leak_v = np.full(n_neuron, leak_v, dtype=np.int32)

        # Vectorize the functions
        gen_ram_frame1_and_2 = np.vectorize(
            _gen_ram_frame1_and_2, otypes=[FRAME_DTYPE, FRAME_DTYPE]
        )
        gen_ram_frame4 = np.vectorize(_gen_ram_frame4, otypes=[FRAME_DTYPE])

        rf1, rf2 = gen_ram_frame1_and_2(leak_v)
        rf4 = gen_ram_frame4(np.arange(n_neuron))

        packages[:, 0] = rf1
        packages[:, 1] = rf2
        packages[:, 2] = rf3
        packages[:, 3] = rf4

        # Tile the package of every neuron `repeat` times & flatten
        # (n_neuron, 4) -> (n_neuron * 4 * repeat,)
        packages_tiled = np.tile(packages, repeat).ravel()

        return packages_tiled


class _OnlineNeuRAMFrame(_NeuRAMFrame):
    def __init__(
        self,
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        neu_start_addr: int,
        n_neuron: int,
        neu_attrs: OnNeuAttrs | dict[str, Any],
        neu_dest_info: OnNeuDestInfo | dict[str, Any],
        weight_width: WeightWidth,
    ) -> None:
        # NOTE: When weight width is 1bit, a neuron need 2 frames (1 address) to store attributes.
        # Otherwise, a neuron need 4 frames (2 addresses) to store attributes.
        if weight_width == WeightWidth.WEIGHT_WIDTH_1BIT:
            self.N_FRAME_PAYLOAD = 2
        else:
            self.N_FRAME_PAYLOAD = 4

        _neu_attrs, _neu_dest_info = _convert_neu_attrs_dest_info(
            neu_attrs, neu_dest_info, "online", weight_width
        )

        super().__init__(
            chip_coord,
            core_coord,
            rid,
            neu_start_addr,
            n_neuron,
            _neu_attrs,
            _neu_dest_info,
            1,
        )

    def get_packages(
        self,
        attrs: dict[str, Any],
        dest_info: dict[str, Any],
        n_neuron: int,
        repeat: int,
    ) -> FrameArrayType:

        def _gen_ram_frame1(idx: int) -> FRAME_DTYPE:
            # Package #1, [63:0]
            return FRAME_DTYPE(
                (
                    (attrs["plasticity_end"] & On_NRAMF.PLASTICITY_END_MASK)
                    << On_NRAMF.PLASTICITY_END_OFFSET
                )
                | (
                    (attrs["plasticity_start"] & On_NRAMF.PLASTICITY_START_MASK)
                    << On_NRAMF.PLASTICITY_START_OFFSET
                )
                | (
                    (addr_axon[idx] & On_NRAMF.ADDR_AXON_MASK)
                    << On_NRAMF.ADDR_AXON_OFFSET
                )
                | (
                    (dest_info["addr_core_y_ex"] & On_NRAMF.ADDR_CORE_Y_EX_MASK)
                    << On_NRAMF.ADDR_CORE_Y_EX_OFFSET
                )
                | (
                    (dest_info["addr_core_x_ex"] & On_NRAMF.ADDR_CORE_X_EX_MASK)
                    << On_NRAMF.ADDR_CORE_X_EX_OFFSET
                )
                | (
                    (dest_info["addr_core_y"] & On_NRAMF.ADDR_CORE_Y_MASK)
                    << On_NRAMF.ADDR_CORE_Y_OFFSET
                )
                | (
                    (dest_info["addr_core_x"] & On_NRAMF.ADDR_CORE_X_MASK)
                    << On_NRAMF.ADDR_CORE_X_OFFSET
                )
                | (
                    (timeslot[idx] & On_NRAMF.TIME_RELATIVE_MASK)
                    << On_NRAMF.TIME_RELATIVE_OFFSET
                )
            )

        def _gen_frame2(leak_v: int, init_v: int) -> FRAME_DTYPE:
            # Package #2, [127:64]
            return FRAME_DTYPE(
                (
                    (attrs["voltage"] & On_NRAMF_WW1.VOLTAGE_MASK)
                    << On_NRAMF_WW1.VOLTAGE_OFFSET
                )
                | ((init_v & On_NRAMF_WW1.INIT_V_MASK) << On_NRAMF_WW1.INIT_V_OFFSET)
                | (
                    (attrs["reset_v"] & On_NRAMF_WW1.RESET_V_MASK)
                    << On_NRAMF_WW1.RESET_V_OFFSET
                )
                | (
                    (attrs["neg_threshold"] & On_NRAMF_WW1.NEG_THRES_MASK)
                    << On_NRAMF_WW1.NEG_THRES_OFFSET
                )
                | (
                    (attrs["pos_threshold"] & On_NRAMF_WW1.POS_THRES_MASK)
                    << On_NRAMF_WW1.POS_THRES_OFFSET
                )
                | ((leak_v & On_NRAMF_WW1.LEAK_V_MASK) << On_NRAMF_WW1.LEAK_V_OFFSET)
            )

        def _gen_frame_2to4(
            leak_v: FRAME_DTYPE, init_v: FRAME_DTYPE
        ) -> tuple[FRAME_DTYPE, FRAME_DTYPE, FRAME_DTYPE]:
            # Package #2, [127:64]
            rf2 = (
                (attrs["voltage"] & On_NRAMF_WWn.VOLTAGE_MASK)
                << On_NRAMF_WWn.VOLTAGE_OFFSET
            ) | ((init_v & On_NRAMF_WWn.INIT_V_MASK) << On_NRAMF_WWn.INIT_V_OFFSET)
            # Package #3, [191:128]
            rf3 = (
                (attrs["reset_v"] & On_NRAMF_WWn.RESET_V_MASK)
                << On_NRAMF_WWn.RESET_V_OFFSET
            ) | (
                (attrs["neg_threshold"] & On_NRAMF_WWn.NEG_THRES_MASK)
                << On_NRAMF_WWn.NEG_THRES_OFFSET
            )
            # Package #4, [255:192]
            rf4 = (
                (attrs["pos_threshold"] & On_NRAMF_WWn.POS_THRES_MASK)
                << On_NRAMF_WWn.POS_THRES_OFFSET
            ) | ((leak_v & On_NRAMF_WWn.LEAK_V_MASK) << On_NRAMF_WWn.LEAK_V_OFFSET)

            return (FRAME_DTYPE(rf2), FRAME_DTYPE(rf3), FRAME_DTYPE(rf4))

        timeslot = dest_info["tick_relative"]
        addr_axon = dest_info["addr_axon"]

        if len(timeslot) != len(addr_axon):
            raise ValueError(
                f"length of 'tick_relative' & 'addr_axon' are not equal, "
                f"{len(timeslot)} != {len(addr_axon)}"
            )

        if n_neuron > len(timeslot):
            raise ValueError(f"length of 'tick_relative' out of range ({n_neuron})")

        packages = np.zeros((n_neuron, self.N_FRAME_PAYLOAD), dtype=FRAME_DTYPE)

        leak_v: int | NDArray[np.int32] = attrs["leak_v"]
        init_v: int | NDArray[np.int32] = attrs["init_v"]

        if isinstance(leak_v, np.ndarray):
            if leak_v.size != n_neuron:
                raise ValueError(
                    f"length of 'leak_v' is not equal to #N of neuron, "
                    f"{leak_v.size} != {n_neuron}"
                )
        else:
            leak_v = np.full(n_neuron, leak_v, dtype=np.int32)

        if isinstance(init_v, np.ndarray):
            if init_v.size != n_neuron:
                raise ValueError(
                    f"length of 'init_v' is not equal to #N of neuron, "
                    f"{init_v.size} != {n_neuron}"
                )
        else:
            init_v = np.full(n_neuron, init_v, dtype=np.int32)

        # Vectorize the functions
        gen_ram_frame1 = np.vectorize(_gen_ram_frame1, otypes=[FRAME_DTYPE])
        gen_frame2 = np.vectorize(_gen_frame2, otypes=[FRAME_DTYPE])
        gen_frame_2to4 = np.vectorize(
            _gen_frame_2to4, otypes=[FRAME_DTYPE, FRAME_DTYPE, FRAME_DTYPE]
        )

        rf1 = gen_ram_frame1(np.arange(n_neuron))

        if self.N_FRAME_PAYLOAD == 2:
            rf_after1 = gen_frame2(leak_v, init_v).reshape(-1, 1)
        else:
            rf_after1 = np.column_stack(gen_frame_2to4(leak_v, init_v))

        packages[:, 0] = rf1
        packages[:, 1:] = rf_after1

        return packages.ravel()


class _WeightRAMFrame(FramePackage, ABC):
    N_WRAM_ADDR: ClassVar[int]
    N_FRAME_PER_WRAM: ClassVar[int]
    N_FRAME_PER_NRAM: ClassVar[int]

    def __init__(
        self,
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        neu_start_addr: int,
        n_package: int,
        weight_ram: FrameArrayType,
    ) -> None:
        if weight_ram.size != n_package:
            raise ValueError(
                f"size of weigh ram must be the #N of packages ({n_package}), but got {weight_ram.size}"
            )

        payload = FramePackagePayload(neu_start_addr, FPType.CONF_TESTOUT, n_package)
        _weight_ram = weight_ram.ravel()

        super().__init__(chip_coord, core_coord, rid, payload, _weight_ram)


class _OfflineWeightRAMFrame(_WeightRAMFrame):
    # Total WRAM: 576Kb
    N_WRAM_ADDR = 512
    N_FRAME_PER_WRAM = 18  # 152 bits per wram address
    N_FRAME_PER_NRAM = (OffCoreParams.ADDR_AXON_MAX + 1) // FF.GENERAL_PACKAGE_LEN


class _OnlineWeightRAMFrame(_WeightRAMFrame):
    # Total WRAM: 1024Kb
    N_WRAM_ADDR = 8192
    N_FRAME_PER_WRAM = 2  # 128 bits per wram address
    N_FRAME_PER_NRAM = (OnCoreParams.ADDR_AXON_MAX + 1) // FF.GENERAL_PACKAGE_LEN


class _LUTRAMFrame(Frame):
    N_FRAME_PER_LUT_RAM: ClassVar[int] = 16

    def __init__(
        self, chip_coord: ChipCoord, core_coord: Coord, rid: RId, lut: LUTDataType
    ) -> None:
        if lut.size != OnCoreParams.LUT_LEN:
            raise ValueError(
                f"size of lut must be {OnCoreParams.LUT_LEN}, but got {lut.size}"
            )

        payload = self.payload_reorganized(lut)
        super().__init__(chip_coord, core_coord, rid, payload)

    @staticmethod
    def payload_reorganized(lut: LUTDataType) -> FrameArrayType:
        lut_u8 = lut.ravel().view(np.uint8).astype(FRAME_DTYPE)
        lut_payload = np.zeros((_LUTRAMFrame.N_FRAME_PER_LUT_RAM,), dtype=FRAME_DTYPE)

        # 15 LUTs per group, 4 groups, 16 frames
        for i in range(4):
            lut_payload[4 * i] = (
                (lut_u8[15 * i] << 22)
                + (lut_u8[15 * i + 1] << 14)
                + (lut_u8[15 * i + 2] << 6)
                + (lut_u8[15 * i + 3] >> 2)  # high6
            )
            lut_payload[4 * i + 1] = (
                ((lut_u8[15 * i + 3] & 0x03) << 28)  # low2
                + (lut_u8[15 * i + 4] << 20)
                + (lut_u8[15 * i + 5] << 12)
                + (lut_u8[15 * i + 6] << 4)
                + (lut_u8[15 * i + 7] >> 4)  # high4
            )
            lut_payload[4 * i + 2] = (
                ((lut_u8[15 * i + 7] & 0x0F) << 26)  # low4
                + (lut_u8[15 * i + 8] << 18)
                + (lut_u8[15 * i + 9] << 10)
                + (lut_u8[15 * i + 10] << 2)
                + (lut_u8[15 * i + 11] >> 6)  # low2
            )
            lut_payload[4 * i + 3] = (
                ((lut_u8[15 * i + 11] & 0x3F) << 24)  # low6
                + (lut_u8[15 * i + 12] << 16)
                + (lut_u8[15 * i + 13] << 8)
                + lut_u8[15 * i + 14]
            )

        return lut_payload.astype(FRAME_DTYPE)


"""Offline Frames"""


class OfflineConfigFrame1(_RandomSeedFrame):
    header = FH.CONFIG_TYPE1


class OfflineConfigFrame2(_OfflineCoreRegFrame):
    header = FH.CONFIG_TYPE2


class OfflineConfigFrame3(_OfflineNeuRAMFrame):
    header = FH.CONFIG_TYPE3


class OfflineConfigFrame4(_OfflineWeightRAMFrame):
    header = FH.CONFIG_TYPE4


class OfflineTestInFrame1(Frame):
    header = FH.TEST_TYPE1

    def __init__(self, chip_coord: ChipCoord, core_coord: Coord, rid: RId) -> None:
        super().__init__(chip_coord, core_coord, rid)


class OfflineTestOutFrame1(_RandomSeedFrame):
    header = FH.TEST_TYPE1


class OfflineTestInFrame2(Frame):
    header = FH.TEST_TYPE2

    def __init__(self, chip_coord: ChipCoord, core_coord: Coord, rid: RId) -> None:
        super().__init__(chip_coord, core_coord, rid)


class OfflineTestOutFrame2(_OfflineCoreRegFrame):
    header = FH.TEST_TYPE2


class OfflineTestInFrame3(FramePackage):
    header = FH.TEST_TYPE3

    def __init__(
        self,
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        neu_start_addr: int,
        n_package: int,
    ) -> None:
        payload = FramePackagePayload(neu_start_addr, FPType.TESTIN, n_package)
        super().__init__(chip_coord, core_coord, rid, payload)


class OfflineTestOutFrame3(_OfflineNeuRAMFrame):
    header = FH.TEST_TYPE3


class OfflineTestInFrame4(FramePackage):
    header = FH.TEST_TYPE4

    def __init__(
        self,
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        neu_start_addr: int,
        n_package: int,
    ) -> None:
        payload = FramePackagePayload(neu_start_addr, FPType.TESTIN, n_package)
        super().__init__(chip_coord, core_coord, rid, payload)


class OfflineTestOutFrame4(_OfflineWeightRAMFrame):
    header = FH.TEST_TYPE4


class OfflineWorkFrame1(Frame):
    header = FH.WORK_TYPE1

    def __init__(
        self,
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        timeslot: IntScalarType,
        axon: IntScalarType,
        data: DataType = 0,  # signed int8
    ) -> None:
        self.ax = int(axon)
        self.ts = int(timeslot)

        if self.ts > Off_WF1F.TIMESLOT_MASK or self.ts < 0:
            raise ValueError(
                f"timeslot out of range {Off_WF1F.TIMESLOT_MASK} ({self.ts})."
            )

        if self.ax > OffCoreParams.ADDR_AXON_MAX or self.ax < 0:
            raise ValueError(
                f"axon out of range {OffCoreParams.ADDR_AXON_MAX} ({self.ax})."
            )

        if isinstance(data, np.ndarray) and data.size != 1:
            raise ShapeError(f"size of data must be 1 ({data.size}).")

        if data < np.iinfo(np.int8).min or data > np.iinfo(np.int8).max:
            raise ValueError(f"data out of range np.int8 ({data}).")

        self.data = PAYLOAD_DATA_DTYPE(data)

        payload = FRAME_DTYPE(
            ((self.ax & Off_WF1F.AXON_MASK) << Off_WF1F.AXON_OFFSET)
            | ((self.ts & Off_WF1F.TIMESLOT_MASK) << Off_WF1F.TIMESLOT_OFFSET)
            | ((data & Off_WF1F.DATA_MASK) << Off_WF1F.DATA_OFFSET)
        )
        super().__init__(chip_coord, core_coord, rid, payload)

    @property
    def target_ts(self) -> int:
        return self.ts

    @property
    def target_axon(self) -> int:
        return self.ax

    @staticmethod
    @params_check(OffNeuDestInfoChecker)
    def _frame_dest_reorganized(dest_info: dict[str, Any]) -> FrameArrayType:
        return OfflineWorkFrame1.concat_frame_dest(
            (dest_info["addr_chip_x"], dest_info["addr_chip_y"]),
            (dest_info["addr_core_x"], dest_info["addr_core_y"]),
            (dest_info["addr_core_x_ex"], dest_info["addr_core_y_ex"]),
            dest_info["addr_axon"],
            dest_info["tick_relative"],
        )

    @classmethod
    def concat_frame_dest(
        cls,
        chip_coord: CoordLike,
        core_coord: CoordLike,
        rid: RIdLike,
        /,
        axons: ArrayLike,
        timeslots: ArrayLike | None = None,
    ) -> FrameArrayType:
        ax = np.asarray(axons, dtype=FRAME_DTYPE).ravel()

        if timeslots is None:
            ts = np.zeros_like(ax)
        else:
            ts = np.asarray(timeslots, dtype=FRAME_DTYPE).ravel()

        if ax.size != ts.size:
            raise ValueError(
                f"the size of axons & timeslots are not equal, {ax.size} != {ts.size}."
            )

        common_head = _get_frame_common(
            cls.header, to_coord(chip_coord), to_coord(core_coord), to_rid(rid)
        )

        payload = ((ax & Off_WF1F.AXON_MASK) << Off_WF1F.AXON_OFFSET) | (
            (ts & Off_WF1F.TIMESLOT_MASK) << Off_WF1F.TIMESLOT_OFFSET
        )

        return (common_head + payload).astype(FRAME_DTYPE)


class _WorkFrame2Base(Frame):
    header = FH.WORK_TYPE2

    def __init__(self, chip_coord: ChipCoord, n_sync: int) -> None:
        if n_sync > Off_WF2F.N_SYNC_MASK:
            warnings.warn(
                OUT_OF_RANGE_WARNING.format("n_sync", 30, n_sync), TruncationWarning
            )

        super().__init__(
            chip_coord,
            Coord(0, 0),
            _rid_unset(),
            FRAME_DTYPE(n_sync & Off_WF2F.N_SYNC_MASK),
        )


class _WorkFrame3Base(Frame):
    header = FH.WORK_TYPE3

    def __init__(self, chip_coord: ChipCoord) -> None:
        super().__init__(chip_coord, Coord(0, 0), _rid_unset())


class _WorkFrame4Base(Frame):
    header = FH.WORK_TYPE4

    def __init__(self, chip_coord: ChipCoord) -> None:
        super().__init__(chip_coord, Coord(0, 0), _rid_unset())


class OfflineWorkFrame2(_WorkFrame2Base):
    pass


class OfflineWorkFrame3(_WorkFrame3Base):
    pass


class OfflineWorkFrame4(_WorkFrame4Base):
    pass


class OnlineConfigFrame1(_LUTRAMFrame):
    header = FH.CONFIG_TYPE1


class OnlineConfigFrame2(_OnlineCoreRegFrame):
    header = FH.CONFIG_TYPE2


class OnlineConfigFrame3(_OnlineNeuRAMFrame):
    header = FH.CONFIG_TYPE3


class OnlineConfigFrame4(_OnlineWeightRAMFrame):
    header = FH.CONFIG_TYPE4


class OnlineTestInFrame1(Frame):
    header = FH.TEST_TYPE1

    def __init__(self, chip_coord: ChipCoord, core_coord: Coord, rid: RId) -> None:
        super().__init__(chip_coord, core_coord, rid)


class OnlineTestOutFrame1(_LUTRAMFrame):
    header = FH.TEST_TYPE1


class OnlineTestInFrame2(Frame):
    header = FH.TEST_TYPE2

    def __init__(self, chip_coord: ChipCoord, core_coord: Coord, rid: RId) -> None:
        super().__init__(chip_coord, core_coord, rid)


class OnlineTestOutFrame2(_OnlineCoreRegFrame):
    header = FH.TEST_TYPE2


class OnlineTestInFrame3(FramePackage):
    header = FH.TEST_TYPE3

    def __init__(
        self,
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        neu_start_addr: int,
        n_package: int,
    ) -> None:
        payload = FramePackagePayload(neu_start_addr, FPType.TESTIN, n_package)
        super().__init__(chip_coord, core_coord, rid, payload)


class OnlineTestOutFrame3(_OnlineNeuRAMFrame):
    header = FH.TEST_TYPE3


class OnlineTestInFrame4(FramePackage):
    header = FH.TEST_TYPE4

    def __init__(
        self,
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        neu_start_addr: int,
        n_package: int,
    ) -> None:
        payload = FramePackagePayload(neu_start_addr, FPType.TESTIN, n_package)
        super().__init__(chip_coord, core_coord, rid, payload)


class OnlineTestOutFrame4(_OnlineWeightRAMFrame):
    header = FH.TEST_TYPE4


class _OnlineWorkFrame1Base(Frame):
    header = FH.WORK_TYPE1

    def __init__(
        self,
        chip_coord: ChipCoord,
        coor_core: Coord,
        rid: RId,
        payload: FRAME_DTYPE = FRAME_DTYPE(0),
    ) -> None:
        if self.subtype is None:
            raise ValueError(f"'subtype' is not set for {self.__class__.__name__}.")

        _payload = FRAME_DTYPE(
            ((self.subtype & On_WF1F.SUBTYPE_MASK) << On_WF1F.SUBTYPE_OFFSET) + payload
        )
        super().__init__(chip_coord, coor_core, rid, _payload, subtype=self.subtype)


class OnlineWorkFrame1_1(_OnlineWorkFrame1Base):
    subtype = Online_WF1F_SubType.TYPE_I

    def __init__(
        self,
        chip_coord: ChipCoord,
        core_coord: Coord,
        rid: RId,
        timeslot: IntScalarType,
        axon: IntScalarType,
    ) -> None:
        self.ax = int(axon)
        self.ts = int(timeslot)

        if self.ts > On_WF1F_1.TIMESLOT_MASK or self.ts < 0:
            raise ValueError(
                f"timeslot out of range {On_WF1F_1.TIMESLOT_MASK} ({self.ts})."
            )

        if self.ax > OnCoreParams.ADDR_AXON_MAX or self.ax < 0:
            raise ValueError(
                f"axon out of range {OnCoreParams.ADDR_AXON_MAX} ({self.ax})."
            )

        payload = FRAME_DTYPE(
            ((axon & On_WF1F_1.AXON_MASK) << On_WF1F_1.AXON_OFFSET)
            | ((timeslot & On_WF1F_1.TIMESLOT_MASK) << On_WF1F_1.TIMESLOT_OFFSET)
        )
        super().__init__(chip_coord, core_coord, rid, payload)

    @property
    def target_ts(self) -> int:
        return self.ts

    @property
    def target_axon(self) -> int:
        return self.ax

    @staticmethod
    @params_check(OnNeuDestInfoChecker)
    def _frame_dest_reorganized(dest_info: dict[str, Any]) -> FrameArrayType:
        return OnlineWorkFrame1_1.concat_frame_dest(
            (dest_info["addr_chip_x"], dest_info["addr_chip_y"]),
            (dest_info["addr_core_x"], dest_info["addr_core_y"]),
            (dest_info["addr_core_x_ex"], dest_info["addr_core_y_ex"]),
            dest_info["addr_axon"],
            dest_info["tick_relative"],
        )

    @classmethod
    def concat_frame_dest(
        cls,
        chip_coord: CoordLike,
        core_coord: CoordLike,
        rid: RIdLike,
        /,
        axons: ArrayLike,
        timeslots: ArrayLike | None = None,
    ) -> FrameArrayType:
        ax = np.asarray(axons, dtype=FRAME_DTYPE).ravel()

        if timeslots is None:
            ts = np.zeros_like(ax)
        else:
            ts = np.asarray(timeslots, dtype=FRAME_DTYPE).ravel()

        if ax.size != ts.size:
            raise ValueError(
                f"the size of axons & timeslots are not equal, {ax.size} != {ts.size}."
            )

        common_head = _get_frame_common(
            cls.header, to_coord(chip_coord), to_coord(core_coord), to_rid(rid)
        )

        if cls.subtype is None:
            raise ValueError(f"'subtype' is not set for {cls.__name__}.")

        payload = (
            ((cls.subtype & On_WF1F_1.SUBTYPE_MASK) << On_WF1F_1.SUBTYPE_OFFSET)
            | ((ax & On_WF1F_1.AXON_MASK) << On_WF1F_1.AXON_OFFSET)
            | ((ts & On_WF1F_1.TIMESLOT_MASK) << On_WF1F_1.TIMESLOT_OFFSET)
        )

        return (common_head + payload).astype(FRAME_DTYPE)


class OnlineWorkFrame1_2(_OnlineWorkFrame1Base):
    subtype = Online_WF1F_SubType.TYPE_II


class OnlineWorkFrame1_3(_OnlineWorkFrame1Base):
    subtype = Online_WF1F_SubType.TYPE_III


class OnlineWorkFrame1_4(_OnlineWorkFrame1Base):
    subtype = Online_WF1F_SubType.TYPE_IV


class OnlineWorkFrame2(_WorkFrame2Base):
    pass


class OnlineWorkFrame3(_WorkFrame3Base):
    pass


class OnlineWorkFrame4(_WorkFrame4Base):
    pass
