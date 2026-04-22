import math
from collections.abc import Sequence
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike

from paicorelib.framelib.frames import NDArray

from ..coordinate import CoordZXYOffset, coordzxy_to_sign_magnitude
from ..core_defs import LCN_EX
from ..core_defs_v2 import CSCAccelerateMode, DataWidth
from ..core_model_v2 import OfflineCoreRegV2, OnlineCoreRegV2
from ..neuron_model_v2 import (
    OfflineNeuDestInfoV2,
    OfflineNeuFoldedAttrsV2Part1,
    OfflineNeuFoldedAttrsV2Part2,
    OfflineNeuFullAttrsV2Part1,
    OfflineNeuFullAttrsV2Part2,
    OfflineNeuHalfAttrsV2,
    OnlineNeuDestInfoV2,
    OnlineNeuFoldedAttrsV2Part1,
    OnlineNeuFoldedAttrsV2Part2,
    OnlineNeuFullAttrsV2Part1,
    OnlineNeuFullAttrsV2Part2,
    OnlineNeuHalfAttrsV2,
)
from ..routing_hexa import AERPacketZXYCopy
from .base import FramePackageHeaderV2, FrameV2, get_frame_dest_v2
from .frame_defs import FrameHeader as FH
from .frame_defs import FramePackageType
from .frame_defs import OfflineConfigFrame1FormatV2 as Off_Cfg1_V2
from .frame_defs import OfflineConfigFrame2FormatV2 as Off_Cfg2_V2
from .frame_defs import OfflineConfigFrame3FormatV2 as Off_Cfg3_V2
from .frame_defs import OfflineControlFrame1FormatV2 as Off_Ctrl1_V2
from .frame_defs import OfflineControlFrame3FormatV2 as Off_Ctrl3_V2
from .frame_defs import OfflineWorkFrame1FormatV2 as Off_Work1_V2
from .frame_defs import OnlineConfigFrame1FormatV2 as On_Cfg1_V2
from .frame_defs import OnlineConfigFrame2FormatV2 as On_Cfg2_V2
from .frame_defs import OnlineConfigFrame3FormatV2 as On_Cfg3_V2
from .frame_defs import OnlineControlFrame1FormatV2 as On_Ctrl1_V2
from .frame_defs import OnlineControlFrame3FormatV2 as On_Ctrl3_V2
from .frame_defs import OnlineControlFrame4FormatV2 as On_Ctrl4_V2
from .frame_defs import OnlineWorkFrame1FormatV2 as On_Work1_V2
from .frame_defs import OnlineWorkFrame2FormatV2 as On_Work2_V2
from .frame_defs import OnlineWorkFrame3FormatV2 as On_Work3_V2
from .frame_defs import OnlineWorkFrame4FormatV2 as On_Work4_V2
from .types import (
    FRAME_DTYPE,
    PAYLOAD_DATA_DTYPE,
    FrameArrayType,
    LUTActivationType,
    LUTPotentialType,
)
from .utils import _mask, bin_split, pack_field

__all__ = ["FrameGenV2", "OfflineFrameGenV2", "OnlineFrameGenV2"]

DataWidthLE8 = Literal[1, 2, 4, 8]
DataWidthLE8Like = DataWidth | DataWidthLE8

_p = pack_field

_UINT_DTYPE_BY_BITS = {
    8: np.uint8,
    16: np.uint16,
    32: np.uint32,
    64: np.uint64,
}
_FLOAT_DTYPE_BY_BITS = {
    16: np.float16,
    32: np.float32,
    64: np.float64,
}


def _normalize_width_le8(
    width: DataWidthLE8Like, *, name: str = "width"
) -> DataWidthLE8:
    width_bits = (1 << width.value) if isinstance(width, DataWidth) else int(width)

    if width_bits not in (1, 2, 4, 8):
        raise ValueError(f"'{name}' only supports 1/2/4/8-bit widths, got {width}.")

    return cast(DataWidthLE8, width_bits)


def _array_to_bits(values: ArrayLike, nbits: Literal[8, 16, 32, 64]) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1)

    if np.issubdtype(arr.dtype, np.floating):
        float_dtype = np.dtype(_FLOAT_DTYPE_BY_BITS[nbits]).newbyteorder("<")
        uint_dtype = np.dtype(_UINT_DTYPE_BY_BITS[nbits]).newbyteorder("<")
        return arr.astype(float_dtype, copy=False).view(uint_dtype)

    return arr.astype(_UINT_DTYPE_BY_BITS[nbits], copy=False)


def _value_to_bits(value: Any, nbits: Literal[8, 16, 32, 64]) -> int:
    return int(_array_to_bits([value], nbits)[0])


def _array_to_bf16_bits(values: ArrayLike) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1)

    if np.issubdtype(arr.dtype, np.floating):
        float_dtype = np.dtype(np.float32).newbyteorder("<")
        uint_dtype = np.dtype(np.uint32).newbyteorder("<")
        arr_f32 = np.ascontiguousarray(arr.astype(float_dtype, copy=False))
        return (arr_f32.view(uint_dtype) >> 16).astype(np.uint16, copy=False)

    return arr.astype(np.uint16, copy=False)


def _value_to_bf16_bits(value: Any) -> int:
    return int(_array_to_bf16_bits([value])[0])


def _normalize_online_lcn(target_lcn: int | LCN_EX) -> int:
    lcn = target_lcn.value if isinstance(target_lcn, LCN_EX) else int(target_lcn)
    if lcn < 0 or lcn >= len(On_Work1_V2.LCN_TO_TS_AXON_WIDTHS):
        raise ValueError(f"'target_lcn' out of range [0, 8], got {target_lcn}.")

    return lcn


def _normalize_online_work_data(data: ArrayLike) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 0:
        arr = arr.reshape(1)

    if arr.dtype.kind == "f":
        if arr.dtype.itemsize not in (2, 4):
            arr = arr.astype(np.float32)
    elif arr.dtype.kind == "b":
        arr = arr.astype(np.uint8, copy=False)
    elif arr.dtype.kind in "iu":
        if arr.dtype.itemsize not in (1, 2, 4):
            if arr.size == 0:
                arr = arr.astype(np.uint8)
            elif arr.dtype.kind == "u":
                vmax = int(arr.max())
                if vmax <= _mask(8):
                    arr = arr.astype(np.uint8)
                elif vmax <= _mask(16):
                    arr = arr.astype(np.uint16)
                else:
                    arr = arr.astype(np.uint32)
            else:
                vmin = int(arr.min())
                vmax = int(arr.max())
                if np.iinfo(np.int8).min <= vmin and vmax <= np.iinfo(np.int8).max:
                    arr = arr.astype(np.int8)
                elif np.iinfo(np.int16).min <= vmin and vmax <= np.iinfo(np.int16).max:
                    arr = arr.astype(np.int16)
                else:
                    arr = arr.astype(np.int32)
    else:
        raise TypeError(f"unsupported dtype for online work data: {arr.dtype}.")

    arr = np.ascontiguousarray(arr.astype(arr.dtype.newbyteorder("<"), copy=False))
    return arr.view(np.uint8).reshape(arr.size, arr.dtype.itemsize)


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
            _p(core_reg["snn_ann"], F.Word1.SNN_ANN_OFFSET, F.Word1.SNN_ANN_MASK)
            | _p(
                core_reg["max_pooling"],
                F.Word1.MAX_POOLING_OFFSET,
                F.Word1.MAX_POOLING_MASK,
            )
            | _p(
                core_reg["add_potential"],
                F.Word1.ADD_POTENTIAL_OFFSET,
                F.Word1.ADD_POTENTIAL_MASK,
            )
            | _p(
                core_reg["zero_output"],
                F.Word1.ZERO_OUTPUT_OFFSET,
                F.Word1.ZERO_OUTPUT_MASK,
            )
            | _p(
                core_reg["input_sign"],
                F.Word1.INPUT_SIGN_OFFSET,
                F.Word1.INPUT_SIGN_MASK,
            )
            | _p(
                core_reg["input_width"],
                F.Word1.INPUT_WIDTH_OFFSET,
                F.Word1.INPUT_WIDTH_MASK,
            )
            | _p(
                core_reg["output_sign"],
                F.Word1.OUTPUT_SIGN_OFFSET,
                F.Word1.OUTPUT_SIGN_MASK,
            )
            | _p(
                core_reg["output_width"],
                F.Word1.OUTPUT_WIDTH_OFFSET,
                F.Word1.OUTPUT_WIDTH_MASK,
            )
            | _p(
                core_reg["weight_sign"],
                F.Word1.WEIGHT_SIGN_OFFSET,
                F.Word1.WEIGHT_SIGN_MASK,
            )
            | _p(
                core_reg["weight_width"],
                F.Word1.WEIGHT_WIDTH_OFFSET,
                F.Word1.WEIGHT_WIDTH_MASK,
            )
            | _p(core_reg["lcn"], F.Word1.LCN_OFFSET, F.Word1.LCN_MASK)
            | _p(
                core_reg["target_lcn"],
                F.Word1.TARGET_LCN_OFFSET,
                F.Word1.TARGET_LCN_MASK,
            )
            | _p(
                core_reg["axon_skew"], F.Word1.AXON_SKEW_OFFSET, F.Word1.AXON_SKEW_MASK
            )
            | _p(
                core_reg["neuron_number"],
                F.Word1.NEURON_NUMBER_OFFSET,
                F.Word1.NEURON_NUMBER_MASK,
            )
            | _p(z, F.Word1.TEST_CORE_XY_OFFSET, F.Word1.TEST_CORE_XY_MASK)
            | _p(x, F.Word1.TEST_CORE_X_OFFSET, F.Word1.TEST_CORE_X_MASK)
            | _p(
                test_core_y_h2,
                F.Word1.TEST_CORE_Y_HIGH2_OFFSET,
                F.Word1.TEST_CORE_Y_HIGH2_MASK,
            )
        )
        w2 = (
            _p(
                test_core_y_l4,
                F.Word2.TEST_CORE_Y_LOW4_OFFSET,
                F.Word2.TEST_CORE_Y_LOW4_MASK,
            )
            | _p(
                core_reg["global_send"],
                F.Word2.GLOBAL_SEND_OFFSET,
                F.Word2.GLOBAL_SEND_MASK,
            )
            | _p(
                core_reg["csc_accelerate"],
                F.Word2.CSC_ACCELERATE_OFFSET,
                F.Word2.CSC_ACCELERATE_MASK,
            )
            | _p(
                core_reg["global_receive"],
                F.Word2.GLOBAL_RECEIVE_OFFSET,
                F.Word2.GLOBAL_RECEIVE_MASK,
            )
            | _p(
                core_reg["thread_number"],
                F.Word2.THREAD_NUMBER_OFFSET,
                F.Word2.THREAD_NUMBER_MASK,
            )
            | _p(
                core_reg["busy_cycle"],
                F.Word2.BUSY_CYCLE_OFFSET,
                F.Word2.BUSY_CYCLE_MASK,
            )
            | _p(
                core_reg["delay_cycle"],
                F.Word2.DELAY_CYCLE_OFFSET,
                F.Word2.DELAY_CYCLE_MASK,
            )
            | _p(
                core_reg["width_cycle"],
                F.Word2.WIDTH_CYCLE_OFFSET,
                F.Word2.WIDTH_CYCLE_MASK,
            )
        )
        w3 = (
            _p(
                core_reg["tick_start"],
                F.Word3.TICK_START_OFFSET,
                F.Word3.TICK_START_MASK,
            )
            | _p(
                core_reg["tick_duration"],
                F.Word3.TICK_DURATION_OFFSET,
                F.Word3.TICK_DURATION_MASK,
            )
            | _p(
                core_reg["tick_initial"],
                F.Word3.TICK_INITIAL_OFFSET,
                F.Word3.TICK_INITIAL_MASK,
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

        if potentials.size != activations.size:
            raise ValueError(
                f"potentials and activations should have the same size, "
                f"but got {potentials.size} != {activations.size}"
            )

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
        folded_attrs2_: list[OfflineNeuFoldedAttrsV2Part2] | list[dict[str, Any]],
    ) -> tuple[FrameArrayType, FrameArrayType, FrameArrayType]:
        """Generate config frame type III neuron packages in aggregate form.

        This function is the aggregate entry for generating three possible package
        groups under config frame type III:
        - half neuron package
        - full neuron package
        - folded neuron package

        For external callers that only need one package, prefer the dedicated
        wrappers:
        - `gen_config_frame3_pkg_half()`
        - `gen_config_frame3_pkg_full()`
        - `gen_config_frame3_pkg_folded()`

        Accepted input combinations:
        - Half only:
          `full_attrs1` is provided, `full_attrs2` is `None`,
          `folded_attrs1` is `None`, `folded_attrs2_` is empty.
        - Full only:
          `full_attrs1` and `full_attrs2` are both provided,
          `folded_attrs1` is `None`, `folded_attrs2_` is empty.
        - Folded only:
          `full_attrs1` and `full_attrs2` are both `None`,
          `folded_attrs1` is provided, `folded_attrs2_` is non-empty.
        - Mixed:
          half/full parameters and folded parameters may be provided together,
          and the function will generate all requested package groups.

        Invalid combinations:
        - `full_attrs2` is provided while `full_attrs1` is missing
        - `folded_attrs1` is provided while `folded_attrs2_` is empty
        - `folded_attrs2_` is non-empty while `folded_attrs1` is missing

        Returns:
            A 3-tuple `(pkg_half_neu, pkg_full_neu, pkg_folded_neu)`.

            - `pkg_half_neu`: encoded half-neuron package frames.
            - `pkg_full_neu`: encoded full-neuron package frames.
            - `pkg_folded_neu`: encoded folded-neuron package frames.

            If a package type is not requested, the corresponding return value is
            an empty `np.ndarray` with dtype `FRAME_DTYPE`.
        """
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
        elif full_attrs2 is not None:
            raise ValueError("attributes of full neuron are incomplete, missing part1")

        if folded_attrs1 is not None:
            if len(folded_attrs2_) == 0:
                raise ValueError(
                    "attributes of folded neuron are incomplete, missing part2"
                )
            else:  # folded
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
        elif len(folded_attrs2_) > 0:
            raise ValueError(
                "attributes of folded neuron are incomplete, missing part1"
            )
        else:
            pass  # empty

        return pkg_half_neu, pkg_full_neu, pkg_folded_neu

    @staticmethod
    def gen_config_frame3_pkg_half(
        dest_info: OfflineNeuDestInfoV2 | dict[str, Any],
        half_attrs: OfflineNeuHalfAttrsV2 | dict[str, Any],
    ) -> FrameArrayType:
        """Generate the half-neuron package of configuration frame type III."""
        pkg_half_neu, _, _ = OfflineFrameGenV2.gen_config_frame3_pkg_neu(
            dest_info, half_attrs, None, None, []
        )
        return pkg_half_neu

    @staticmethod
    def gen_config_frame3_pkg_full(
        dest_info: OfflineNeuDestInfoV2 | dict[str, Any],
        full_attrs1: OfflineNeuFullAttrsV2Part1 | dict[str, Any],
        full_attrs2: OfflineNeuFullAttrsV2Part2 | dict[str, Any],
    ) -> FrameArrayType:
        """Generate the full-neuron package of configuration frame type III."""
        _, pkg_full_neu, _ = OfflineFrameGenV2.gen_config_frame3_pkg_neu(
            dest_info, full_attrs1, full_attrs2, None, []
        )
        return pkg_full_neu

    @staticmethod
    def gen_config_frame3_pkg_folded(
        folded_attrs1: OfflineNeuFoldedAttrsV2Part1 | dict[str, Any],
        folded_attrs2_: Sequence[OfflineNeuFoldedAttrsV2Part2 | dict[str, Any]],
    ) -> FrameArrayType:
        """Generate the folded-neuron package of configuration frame type III."""
        if len(folded_attrs2_) == 0:
            raise ValueError("attributes of folded neuron are incomplete")

        folded_attrs1 = OfflineNeuFoldedAttrsV2Part1.model_validate(
            folded_attrs1, strict=True
        ).model_dump()
        folded_attrs2 = [
            OfflineNeuFoldedAttrsV2Part2.model_validate(
                attrs2, strict=True
            ).model_dump()
            for attrs2 in folded_attrs2_
        ]
        return OfflineFrameGenV2._gen_pkg_folded_neu(folded_attrs1, *folded_attrs2)

    @staticmethod
    def _gen_pkg_half_neu(
        dest_info: dict[str, Any], half_attrs: dict[str, Any]
    ) -> FrameArrayType:
        F = Off_Cfg3_V2.Full
        weight_skew_h11, weight_skew_l5 = bin_split(half_attrs["weight_skew"], 5, 11)
        # RAM[0][63:0]
        w1 = (
            _p(
                weight_skew_l5,
                F.Word1.WEIGHT_SKEW_LOW5_OFFSET,
                F.Word1.WEIGHT_SKEW_LOW5_MASK,
            )
            | _p(
                half_attrs["weight_address_start"],
                F.Word1.WEIGHT_ADDRESS_START_OFFSET,
                F.Word1.WEIGHT_ADDRESS_START_MASK,
            )
            | _p(
                half_attrs["weight_address_end"],
                F.Word1.WEIGHT_ADDRESS_END_OFFSET,
                F.Word1.WEIGHT_ADDRESS_END_MASK,
            )
            | _p(
                half_attrs["output_type"],
                F.Word1.OUTPUT_TYPE_OFFSET,
                F.Word1.OUTPUT_TYPE_MASK,
            )
            | _p(
                half_attrs["fold_type"],
                F.Word1.FOLD_TYPE_OFFSET,
                F.Word1.FOLD_TYPE_MASK,
            )
            | _p(
                half_attrs["neuron_type"],
                F.Word1.NEURON_TYPE_OFFSET,
                F.Word1.NEURON_TYPE_MASK,
            )
            | _p(half_attrs["vjt"], F.Word1.VJT_OFFSET, F.Word1.VJT_MASK)
        )
        # RAM[0][127:64]
        w2 = (
            _p(
                dest_info["tick_relative"],
                F.Word2.TICK_RELATIVE_OFFSET,
                F.Word2.TICK_RELATIVE_MASK,
            )
            | _p(
                dest_info["addr_axon"], F.Word2.ADDR_AXON_OFFSET, F.Word2.ADDR_AXON_MASK
            )
            | _p(
                dest_info["addr_core_xy"],
                F.Word2.ADDR_CORE_XY_OFFSET,
                F.Word2.ADDR_CORE_XY_MASK,
            )
            | _p(
                dest_info["addr_core_x"],
                F.Word2.ADDR_CORE_X_OFFSET,
                F.Word2.ADDR_CORE_X_MASK,
            )
            | _p(
                dest_info["addr_core_y"],
                F.Word2.ADDR_CORE_Y_OFFSET,
                F.Word2.ADDR_CORE_Y_MASK,
            )
            | _p(
                dest_info["addr_copy_xy"],
                F.Word2.ADDR_COPY_XY_OFFSET,
                F.Word2.ADDR_COPY_XY_MASK,
            )
            | _p(
                dest_info["addr_copy_x"],
                F.Word2.ADDR_COPY_X_OFFSET,
                F.Word2.ADDR_COPY_X_MASK,
            )
            | _p(
                dest_info["addr_copy_y"],
                F.Word2.ADDR_COPY_Y_OFFSET,
                F.Word2.ADDR_COPY_Y_MASK,
            )
            | _p(
                weight_skew_h11,
                F.Word2.WEIGHT_SKEW_HIGH11_OFFSET,
                F.Word2.WEIGHT_SKEW_HIGH11_MASK,
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
        thres_pos_h12, thres_pos_l20 = bin_split(full_attrs2["threshold_pos"], 20, 12)
        # RAM[1][63:0]
        w3 = (
            _p(
                thres_pos_l20,
                F.Word3.THRESHOLD_POS_LOW20_OFFSET,
                F.Word3.THRESHOLD_POS_LOW20_MASK,
            )
            | _p(
                full_attrs2["lateral_inhibition"],
                F.Word3.LATERAL_INHIBITION_OFFSET,
                F.Word3.LATERAL_INHIBITION_MASK,
            )
            | _p(
                full_attrs2["leak_multi_sequence"],
                F.Word3.LEAK_MULTI_SEQUENCE_OFFSET,
                F.Word3.LEAK_MULTI_SEQUENCE_MASK,
            )
            | _p(
                full_attrs2["leak_multi_input"],
                F.Word3.LEAK_MULTI_INPUT_OFFSET,
                F.Word3.LEAK_MULTI_INPUT_MASK,
            )
            | _p(
                full_attrs2["leak_multi_mode"],
                F.Word3.LEAK_MULTI_MODE_OFFSET,
                F.Word3.LEAK_MULTI_MODE_MASK,
            )
            | _p(
                full_attrs2["leak_add_mode"],
                F.Word3.LEAK_ADD_MODE_OFFSET,
                F.Word3.LEAK_ADD_MODE_MASK,
            )
            | _p(
                full_attrs2["leak_tau"], F.Word3.LEAK_TAU_OFFSET, F.Word3.LEAK_TAU_MASK
            )
            | _p(full_attrs2["leak_v"], F.Word3.LEAK_V_OFFSET, F.Word3.LEAK_V_MASK)
            | _p(
                full_attrs2["weight_compress"],
                F.Word3.WEIGHT_COMPRESS_OFFSET,
                F.Word3.WEIGHT_COMPRESS_MASK,
            )
            | _p(
                full_attrs2["vjt_initial"],
                F.Word3.VJT_INITIAL_OFFSET,
                F.Word3.VJT_INITIAL_MASK,
            )
        )
        # RAM[1][127:64]
        w4 = (
            _p(
                full_attrs2["reset_mode"],
                F.Word4.RESET_MODE_OFFSET,
                F.Word4.RESET_MODE_MASK,
            )
            | _p(full_attrs2["reset_v"], F.Word4.RESET_V_OFFSET, F.Word4.RESET_V_MASK)
            | _p(
                full_attrs2["threshold_neg_mode"],
                F.Word4.THRESHOLD_NEG_MODE_OFFSET,
                F.Word4.THRESHOLD_NEG_MODE_MASK,
            )
            | _p(
                full_attrs2["threshold_pos_mode"],
                F.Word4.THRESHOLD_POS_MODE_OFFSET,
                F.Word4.THRESHOLD_POS_MODE_MASK,
            )
            | _p(
                full_attrs2["threshold_neg"],
                F.Word4.THRESHOLD_NEG_OFFSET,
                F.Word4.THRESHOLD_NEG_MASK,
            )
            | _p(
                thres_pos_h12,
                F.Word4.THRESHOLD_POS_HIGH12_OFFSET,
                F.Word4.THRESHOLD_POS_HIGH12_MASK,
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
            _p(
                fold_skew_y_l2,
                F.Word1.FOLD_SKEW_Y_LOW2_OFFSET,
                F.Word1.FOLD_SKEW_Y_LOW2_MASK,
            )
            | _p(
                folded_attrs1["fold_axon_xy"],
                F.Word1.FOLD_AXON_XY_OFFSET,
                F.Word1.FOLD_AXON_XY_MASK,
            )
            | _p(
                folded_attrs1["fold_axon_x"],
                F.Word1.FOLD_AXON_X_OFFSET,
                F.Word1.FOLD_AXON_X_MASK,
            )
            | _p(
                folded_attrs1["fold_axon_y"],
                F.Word1.FOLD_AXON_Y_OFFSET,
                F.Word1.FOLD_AXON_Y_MASK,
            )
            | _p(
                folded_attrs1["fold_number"],
                F.Word1.FOLD_NUMBER_OFFSET,
                F.Word1.FOLD_NUMBER_MASK,
            )
        )
        # RAM[1][127:64]
        w2 = (
            _p(
                folded_attrs1["fold_range_xy"],
                F.Word2.FOLD_RANGE_XY_OFFSET,
                F.Word2.FOLD_RANGE_XY_MASK,
            )
            | _p(
                folded_attrs1["fold_range_x"],
                F.Word2.FOLD_RANGE_X_OFFSET,
                F.Word2.FOLD_RANGE_X_MASK,
            )
            | _p(
                folded_attrs1["fold_range_y"],
                F.Word2.FOLD_RANGE_Y_OFFSET,
                F.Word2.FOLD_RANGE_Y_MASK,
            )
            | _p(
                folded_attrs1["fold_skew_xy"],
                F.Word2.FOLD_SKEW_XY_OFFSET,
                F.Word2.FOLD_SKEW_XY_MASK,
            )
            | _p(
                folded_attrs1["fold_skew_x"],
                F.Word2.FOLD_SKEW_X_OFFSET,
                F.Word2.FOLD_SKEW_X_MASK,
            )
            | _p(
                fold_skew_y_h9,
                F.Word2.FOLD_SKEW_Y_HIGH9_OFFSET,
                F.Word2.FOLD_SKEW_Y_HIGH9_MASK,
            )
        )

        if len(folded_attrs2) == 0:
            raise ValueError("at least one folded neuron attrs part2 is required")

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
        weight_width: DataWidthLE8Like,
        input_width: DataWidthLE8Like,
        csc_compress: bool | CSCAccelerateMode = False,
    ) -> FrameArrayType:
        """Generate weight package for config frame type III."""
        if weight.ndim != 1:
            raise ValueError(
                f"'weight' must be a 1D array, but got ndim={weight.ndim}."
            )

        is_compress = csc_compress != CSCAccelerateMode.DISABLE
        norm_weight_width = _normalize_width_le8(weight_width, name="weight_width")
        norm_input_width = _normalize_width_le8(input_width, name="input_width")

        if is_compress:
            return weight_csc_pack(weight, norm_weight_width, norm_input_width)
        else:
            return weight_dense_pack(weight, norm_weight_width)

    @staticmethod
    def gen_config_frame4(
        pkt_offset: CoordZXYOffset,
        input_array: FrameArrayType,
        pkt_ncopy: AERPacketZXYCopy = AERPacketZXYCopy(),
        start_addr: int = 0,
    ):
        raise NotImplementedError

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
            _p(ts_msb, F.TIMESTEP_HIGH7_OFFSET, F.TIMESTEP_HIGH7_MASK)
            | (ts_low << (F.AXON_ADDR_OFFSET + AX_WIDTH))
            | ((ax & _mask(AX_WIDTH)) << F.AXON_ADDR_OFFSET)
        )

        mask = np.flatnonzero(_data)
        data_u8 = _data.astype(PAYLOAD_DATA_DTYPE)
        frame_dest = get_frame_dest_v2(FH.WORK_TYPE1, pkt_offset, pkt_ncopy)
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


class OnlineFrameGenV2(FrameGenV2):
    LCN_TO_TS_AXON_WIDTHS = On_Work1_V2.LCN_TO_TS_AXON_WIDTHS

    @staticmethod
    def gen_config_frame1(
        pkt_offset: CoordZXYOffset,
        core_reg_: OnlineCoreRegV2 | dict[str, Any],
        pkt_ncopy: AERPacketZXYCopy = AERPacketZXYCopy(),
        start_addr: int = 0,
    ) -> FrameArrayType:
        F = On_Cfg1_V2
        core_reg = OnlineCoreRegV2.model_validate(core_reg_, strict=True).model_dump()

        neuron_number_h10, neuron_number_l3 = bin_split(
            int(core_reg["neuron_number"]), 3, 10
        )
        scale_out_bits = _value_to_bf16_bits(core_reg["scale_out"])
        scale_out_h15, scale_out_l1 = bin_split(scale_out_bits, 1, 15)
        update_core_xy, update_core_x, update_core_y = coordzxy_to_sign_magnitude(
            (
                int(core_reg["update_core_xy"]),
                int(core_reg["update_core_x"]),
                int(core_reg["update_core_y"]),
            )
        )
        test_core_xy, test_core_x, test_core_y = coordzxy_to_sign_magnitude(
            (
                int(core_reg["test_core_xy"]),
                int(core_reg["test_core_x"]),
                int(core_reg["test_core_y"]),
            )
        )
        test_core_y_h1, test_core_y_l5 = bin_split(test_core_y, 5, 1)

        w1 = (
            _p(core_reg["snn_ann"], F.Word1.SNN_ANN_OFFSET, F.Word1.SNN_ANN_MASK)
            | _p(
                core_reg["max_pooling"],
                F.Word1.MAX_POOLING_OFFSET,
                F.Word1.MAX_POOLING_MASK,
            )
            | _p(
                core_reg["add_potential"],
                F.Word1.ADD_POTENTIAL_OFFSET,
                F.Word1.ADD_POTENTIAL_MASK,
            )
            | _p(
                core_reg["zero_output"],
                F.Word1.ZERO_OUTPUT_OFFSET,
                F.Word1.ZERO_OUTPUT_MASK,
            )
            | _p(
                core_reg["work_mode"], F.Word1.WORK_MODE_OFFSET, F.Word1.WORK_MODE_MASK
            )
            | _p(
                core_reg["input_core"],
                F.Word1.INPUT_CORE_OFFSET,
                F.Word1.INPUT_CORE_MASK,
            )
            | _p(
                core_reg["input_width"],
                F.Word1.INPUT_WIDTH_OFFSET,
                F.Word1.INPUT_WIDTH_MASK,
            )
            | _p(
                core_reg["output_core"],
                F.Word1.OUTPUT_CORE_OFFSET,
                F.Word1.OUTPUT_CORE_MASK,
            )
            | _p(
                core_reg["output_width"],
                F.Word1.OUTPUT_WIDTH_OFFSET,
                F.Word1.OUTPUT_WIDTH_MASK,
            )
            | _p(core_reg["LCN_AT"], F.Word1.LCN_AT_OFFSET, F.Word1.LCN_AT_MASK)
            | _p(core_reg["LCN_MP"], F.Word1.LCN_MP_OFFSET, F.Word1.LCN_MP_MASK)
            | _p(core_reg["LCN_LG"], F.Word1.LCN_LG_OFFSET, F.Word1.LCN_LG_MASK)
            | _p(
                core_reg["target_LCN_AT"],
                F.Word1.TARGET_LCN_AT_OFFSET,
                F.Word1.TARGET_LCN_AT_MASK,
            )
            | _p(
                core_reg["target_LCN_MP"],
                F.Word1.TARGET_LCN_MP_OFFSET,
                F.Word1.TARGET_LCN_MP_MASK,
            )
            | _p(
                core_reg["target_LCN_LG"],
                F.Word1.TARGET_LCN_LG_OFFSET,
                F.Word1.TARGET_LCN_LG_MASK,
            )
            | _p(
                core_reg["axon_skew"], F.Word1.AXON_SKEW_OFFSET, F.Word1.AXON_SKEW_MASK
            )
            | _p(
                neuron_number_h10,
                F.Word1.NEURON_NUMBER_HIGH10_OFFSET,
                F.Word1.NEURON_NUMBER_HIGH10_MASK,
            )
        )
        w2 = (
            _p(
                neuron_number_l3,
                F.Word2.NEURON_NUMBER_LOW3_OFFSET,
                F.Word2.NEURON_NUMBER_LOW3_MASK,
            )
            | _p(
                core_reg["update_number"],
                F.Word2.UPDATE_NUMBER_OFFSET,
                F.Word2.UPDATE_NUMBER_MASK,
            )
            | _p(
                core_reg["csc_accelerate"],
                F.Word2.CSC_ACCELERATE_OFFSET,
                F.Word2.CSC_ACCELERATE_MASK,
            )
            | _p(
                _value_to_bf16_bits(core_reg["scale_in"]),
                F.Word2.SCALE_IN_OFFSET,
                F.Word2.SCALE_IN_MASK,
            )
            | _p(
                _value_to_bf16_bits(core_reg["bias_in"]),
                F.Word2.BIAS_IN_OFFSET,
                F.Word2.BIAS_IN_MASK,
            )
            | _p(
                scale_out_h15,
                F.Word2.SCALE_OUT_HIGH15_OFFSET,
                F.Word2.SCALE_OUT_HIGH15_MASK,
            )
        )
        w3 = (
            _p(
                scale_out_l1,
                F.Word3.SCALE_OUT_LOW1_OFFSET,
                F.Word3.SCALE_OUT_LOW1_MASK,
            )
            | _p(
                _value_to_bf16_bits(core_reg["bias_out"]),
                F.Word3.BIAS_OUT_OFFSET,
                F.Word3.BIAS_OUT_MASK,
            )
            | _p(
                _value_to_bf16_bits(core_reg["learning_rate"]),
                F.Word3.LEARNING_RATE_OFFSET,
                F.Word3.LEARNING_RATE_MASK,
            )
            | _p(
                update_core_xy,
                F.Word3.UPDATE_CORE_XY_OFFSET,
                F.Word3.UPDATE_CORE_XY_MASK,
            )
            | _p(
                update_core_x,
                F.Word3.UPDATE_CORE_X_OFFSET,
                F.Word3.UPDATE_CORE_X_MASK,
            )
            | _p(
                update_core_y,
                F.Word3.UPDATE_CORE_Y_OFFSET,
                F.Word3.UPDATE_CORE_Y_MASK,
            )
            | _p(
                test_core_xy,
                F.Word3.TEST_CORE_XY_OFFSET,
                F.Word3.TEST_CORE_XY_MASK,
            )
            | _p(
                test_core_x,
                F.Word3.TEST_CORE_X_OFFSET,
                F.Word3.TEST_CORE_X_MASK,
            )
            | _p(
                test_core_y_h1,
                F.Word3.TEST_CORE_Y_HIGH1_OFFSET,
                F.Word3.TEST_CORE_Y_HIGH1_MASK,
            )
        )
        w4 = (
            _p(
                test_core_y_l5,
                F.Word4.TEST_CORE_Y_LOW5_OFFSET,
                F.Word4.TEST_CORE_Y_LOW5_MASK,
            )
            | _p(
                core_reg["global_send"],
                F.Word4.GLOBAL_SEND_OFFSET,
                F.Word4.GLOBAL_SEND_MASK,
            )
            | _p(
                core_reg["global_receive"],
                F.Word4.GLOBAL_RECEIVE_OFFSET,
                F.Word4.GLOBAL_RECEIVE_MASK,
            )
            | _p(
                core_reg["thread_number"],
                F.Word4.THREAD_NUMBER_OFFSET,
                F.Word4.THREAD_NUMBER_MASK,
            )
            | _p(
                core_reg["busy_cycle"],
                F.Word4.BUSY_CYCLE_OFFSET,
                F.Word4.BUSY_CYCLE_MASK,
            )
            | _p(
                core_reg["delay_cycle"],
                F.Word4.DELAY_CYCLE_OFFSET,
                F.Word4.DELAY_CYCLE_MASK,
            )
            | _p(
                core_reg["width_cycle"],
                F.Word4.WIDTH_CYCLE_OFFSET,
                F.Word4.WIDTH_CYCLE_MASK,
            )
        )
        w5 = (
            _p(
                core_reg["tick_start"],
                F.Word5.TICK_START_OFFSET,
                F.Word5.TICK_START_MASK,
            )
            | _p(
                core_reg["tick_duration"],
                F.Word5.TICK_DURATION_OFFSET,
                F.Word5.TICK_DURATION_MASK,
            )
            | _p(
                core_reg["tick_initial"],
                F.Word5.TICK_INITIAL_OFFSET,
                F.Word5.TICK_INITIAL_MASK,
            )
        )

        pkg = np.array([w1, w2, w3, w4, w5], dtype=FRAME_DTYPE)
        return OnlineFrameGenV2.make_package(
            FH.CONFIG_TYPE1, pkt_offset, pkt_ncopy, start_addr, pkg
        )

    @staticmethod
    def gen_config_frame2(
        pkt_offset: CoordZXYOffset,
        potentials: ArrayLike,
        activations: ArrayLike,
        pkt_ncopy: AERPacketZXYCopy = AERPacketZXYCopy(),
        start_addr: int = 0,
    ) -> FrameArrayType:
        F = On_Cfg2_V2
        arr_pot_u64 = _array_to_bits(potentials, 32).astype(FRAME_DTYPE).ravel()
        arr_act_u64 = _array_to_bf16_bits(activations).astype(FRAME_DTYPE).ravel()

        if arr_pot_u64.size != arr_act_u64.size:
            raise ValueError(
                f"potentials and activations should have the same size, "
                f"but got {arr_pot_u64.size} != {arr_act_u64.size}"
            )

        packages = (
            _p(arr_pot_u64, F.POTENTIAL_OFFSET, F.POTENTIAL_MASK)
            | _p(arr_act_u64, F.ACTIVATION_OFFSET, F.ACTIVATION_MASK)
        ).astype(FRAME_DTYPE)
        return OnlineFrameGenV2.make_package(
            FH.CONFIG_TYPE2, pkt_offset, pkt_ncopy, start_addr, packages
        )

    @staticmethod
    def gen_config_frame3_pkg_header(
        pkt_offset: CoordZXYOffset,
        start_addr: int,
        n_package: int,
        pkt_ncopy: AERPacketZXYCopy = AERPacketZXYCopy(),
    ) -> FrameArrayType:
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
        dest_info: OnlineNeuDestInfoV2 | dict[str, Any],
        full_attrs1: OnlineNeuFullAttrsV2Part1 | dict[str, Any] | None,
        full_attrs2: OnlineNeuFullAttrsV2Part2 | dict[str, Any] | None,
        folded_attrs1: OnlineNeuFoldedAttrsV2Part1 | dict[str, Any] | None,
        folded_attrs2_: Sequence[OnlineNeuFoldedAttrsV2Part2 | dict[str, Any]],
    ) -> tuple[FrameArrayType, FrameArrayType, FrameArrayType]:
        dest_info_dump = OnlineNeuDestInfoV2.model_validate(
            dest_info, strict=True
        ).model_dump()
        pkg_half_neu = np.array([], dtype=FRAME_DTYPE)
        pkg_full_neu = np.array([], dtype=FRAME_DTYPE)
        pkg_folded_neu = np.array([], dtype=FRAME_DTYPE)

        if full_attrs1 is not None:
            full_attrs1_dump = OnlineNeuFullAttrsV2Part1.model_validate(
                full_attrs1, strict=True
            ).model_dump()
            pkg_half_neu = OnlineFrameGenV2._gen_pkg_half_neu(
                dest_info_dump, full_attrs1_dump
            )

            if full_attrs2 is not None:
                full_attrs2_dump = OnlineNeuFullAttrsV2Part2.model_validate(
                    full_attrs2, strict=True
                ).model_dump()
                pkg_full_neu = OnlineFrameGenV2._gen_pkg_full_neu(
                    dest_info_dump, full_attrs1_dump, full_attrs2_dump
                )
        elif full_attrs2 is not None:
            raise ValueError("attributes of full neuron are incomplete, missing part1")

        if folded_attrs1 is not None:
            if len(folded_attrs2_) == 0:
                raise ValueError(
                    "attributes of folded neuron are incomplete, missing part2"
                )

            folded_attrs1_dump = OnlineNeuFoldedAttrsV2Part1.model_validate(
                folded_attrs1, strict=True
            ).model_dump()
            folded_attrs2_dump = [
                OnlineNeuFoldedAttrsV2Part2.model_validate(
                    attrs2, strict=True
                ).model_dump()
                for attrs2 in folded_attrs2_
            ]
            pkg_folded_neu = OnlineFrameGenV2._gen_pkg_folded_neu(
                folded_attrs1_dump, *folded_attrs2_dump
            )
        elif len(folded_attrs2_) > 0:
            raise ValueError(
                "attributes of folded neuron are incomplete, missing part1"
            )

        return pkg_half_neu, pkg_full_neu, pkg_folded_neu

    @staticmethod
    def gen_config_frame3_pkg_half(
        dest_info: OnlineNeuDestInfoV2 | dict[str, Any],
        half_attrs: OnlineNeuHalfAttrsV2 | dict[str, Any],
    ) -> FrameArrayType:
        pkg_half_neu, _, _ = OnlineFrameGenV2.gen_config_frame3_pkg_neu(
            dest_info, half_attrs, None, None, []
        )
        return pkg_half_neu

    @staticmethod
    def gen_config_frame3_pkg_full(
        dest_info: OnlineNeuDestInfoV2 | dict[str, Any],
        full_attrs1: OnlineNeuFullAttrsV2Part1 | dict[str, Any],
        full_attrs2: OnlineNeuFullAttrsV2Part2 | dict[str, Any],
    ) -> FrameArrayType:
        _, pkg_full_neu, _ = OnlineFrameGenV2.gen_config_frame3_pkg_neu(
            dest_info, full_attrs1, full_attrs2, None, []
        )
        return pkg_full_neu

    @staticmethod
    def gen_config_frame3_pkg_folded(
        folded_attrs1: OnlineNeuFoldedAttrsV2Part1 | dict[str, Any],
        folded_attrs2_: Sequence[OnlineNeuFoldedAttrsV2Part2 | dict[str, Any]],
    ) -> FrameArrayType:
        if len(folded_attrs2_) == 0:
            raise ValueError("attributes of folded neuron are incomplete")

        folded_attrs1_dump = OnlineNeuFoldedAttrsV2Part1.model_validate(
            folded_attrs1, strict=True
        ).model_dump()
        folded_attrs2_dump = [
            OnlineNeuFoldedAttrsV2Part2.model_validate(attrs2, strict=True).model_dump()
            for attrs2 in folded_attrs2_
        ]
        return OnlineFrameGenV2._gen_pkg_folded_neu(
            folded_attrs1_dump, *folded_attrs2_dump
        )

    @staticmethod
    def _gen_pkg_half_neu(
        dest_info: dict[str, Any], half_attrs: dict[str, Any]
    ) -> FrameArrayType:
        F = On_Cfg3_V2.Full
        weight_skew_h12, weight_skew_l4 = bin_split(
            int(half_attrs["weight_skew"]), 4, 12
        )
        addr_core_xy, addr_core_x, addr_core_y = coordzxy_to_sign_magnitude(
            (
                int(dest_info["addr_core_xy"]),
                int(dest_info["addr_core_x"]),
                int(dest_info["addr_core_y"]),
            )
        )
        addr_copy_xy, addr_copy_x, addr_copy_y = coordzxy_to_sign_magnitude(
            (
                int(dest_info["addr_copy_xy"]),
                int(dest_info["addr_copy_x"]),
                int(dest_info["addr_copy_y"]),
            )
        )
        # Online config frame type III is packed from RAM[n][63:0] to RAM[n][127:64].
        w1 = (
            _p(
                weight_skew_l4,
                F.Word1.WEIGHT_SKEW_LOW4_OFFSET,
                F.Word1.WEIGHT_SKEW_LOW4_MASK,
            )
            | _p(
                half_attrs["weight_address_start"],
                F.Word1.WEIGHT_ADDRESS_START_OFFSET,
                F.Word1.WEIGHT_ADDRESS_START_MASK,
            )
            | _p(
                half_attrs["weight_address_end"],
                F.Word1.WEIGHT_ADDRESS_END_OFFSET,
                F.Word1.WEIGHT_ADDRESS_END_MASK,
            )
            | _p(
                half_attrs["output_type"],
                F.Word1.OUTPUT_TYPE_OFFSET,
                F.Word1.OUTPUT_TYPE_MASK,
            )
            | _p(
                half_attrs["fold_type"],
                F.Word1.FOLD_TYPE_OFFSET,
                F.Word1.FOLD_TYPE_MASK,
            )
            | _p(
                half_attrs["neuron_type"],
                F.Word1.NEURON_TYPE_OFFSET,
                F.Word1.NEURON_TYPE_MASK,
            )
            | _p(
                _value_to_bits(half_attrs["vjt"], 32),
                F.Word1.VJT_OFFSET,
                F.Word1.VJT_MASK,
            )
        )
        w2 = (
            _p(
                dest_info["tick_relative"],
                F.Word2.TICK_RELATIVE_OFFSET,
                F.Word2.TICK_RELATIVE_MASK,
            )
            | _p(
                dest_info["addr_axon"],
                F.Word2.ADDR_AXON_OFFSET,
                F.Word2.ADDR_AXON_MASK,
            )
            | _p(
                addr_core_xy,
                F.Word2.ADDR_CORE_XY_OFFSET,
                F.Word2.ADDR_CORE_XY_MASK,
            )
            | _p(
                addr_core_x,
                F.Word2.ADDR_CORE_X_OFFSET,
                F.Word2.ADDR_CORE_X_MASK,
            )
            | _p(
                addr_core_y,
                F.Word2.ADDR_CORE_Y_OFFSET,
                F.Word2.ADDR_CORE_Y_MASK,
            )
            | _p(
                addr_copy_xy,
                F.Word2.ADDR_COPY_XY_OFFSET,
                F.Word2.ADDR_COPY_XY_MASK,
            )
            | _p(
                addr_copy_x,
                F.Word2.ADDR_COPY_X_OFFSET,
                F.Word2.ADDR_COPY_X_MASK,
            )
            | _p(
                addr_copy_y,
                F.Word2.ADDR_COPY_Y_OFFSET,
                F.Word2.ADDR_COPY_Y_MASK,
            )
            | _p(
                weight_skew_h12,
                F.Word2.WEIGHT_SKEW_HIGH12_OFFSET,
                F.Word2.WEIGHT_SKEW_HIGH12_MASK,
            )
        )
        return np.array([w1, w2], dtype=FRAME_DTYPE)

    @staticmethod
    def _gen_pkg_full_neu(
        dest_info: dict[str, Any],
        full_attrs1: dict[str, Any],
        full_attrs2: dict[str, Any],
    ) -> FrameArrayType:
        F = On_Cfg3_V2.Full
        pkg_half_neu = OnlineFrameGenV2._gen_pkg_half_neu(dest_info, full_attrs1)
        threshold_pos_h12, threshold_pos_l20 = bin_split(
            _value_to_bits(full_attrs2["threshold_pos"], 32), 20, 12
        )
        # Continue the same little-endian RAM word order: low 64 bits first, then high 64 bits.
        w3 = (
            _p(
                threshold_pos_l20,
                F.Word3.THRESHOLD_POS_LOW20_OFFSET,
                F.Word3.THRESHOLD_POS_LOW20_MASK,
            )
            | _p(
                full_attrs2["lateral_inhibition"],
                F.Word3.LATERAL_INHIBITION_OFFSET,
                F.Word3.LATERAL_INHIBITION_MASK,
            )
            | _p(
                full_attrs2["leak_multi_sequence"],
                F.Word3.LEAK_MULTI_SEQUENCE_OFFSET,
                F.Word3.LEAK_MULTI_SEQUENCE_MASK,
            )
            | _p(
                full_attrs2["leak_multi_input"],
                F.Word3.LEAK_MULTI_INPUT_OFFSET,
                F.Word3.LEAK_MULTI_INPUT_MASK,
            )
            | _p(
                full_attrs2["leak_multi_mode"],
                F.Word3.LEAK_MULTI_MODE_OFFSET,
                F.Word3.LEAK_MULTI_MODE_MASK,
            )
            | _p(
                full_attrs2["leak_add_mode"],
                F.Word3.LEAK_ADD_MODE_OFFSET,
                F.Word3.LEAK_ADD_MODE_MASK,
            )
            | _p(
                full_attrs2["leak_tau"],
                F.Word3.LEAK_TAU_OFFSET,
                F.Word3.LEAK_TAU_MASK,
            )
            | _p(
                _value_to_bf16_bits(full_attrs2["vjt_initial"]),
                F.Word3.VJT_INITIAL_OFFSET,
                F.Word3.VJT_INITIAL_MASK,
            )
            | _p(
                full_attrs2["weight_compress"],
                F.Word3.WEIGHT_COMPRESS_OFFSET,
                F.Word3.WEIGHT_COMPRESS_MASK,
            )
            | _p(
                _value_to_bf16_bits(full_attrs2["leak_v"]),
                F.Word3.LEAK_V_OFFSET,
                F.Word3.LEAK_V_MASK,
            )
        )
        w4 = (
            _p(
                getattr(full_attrs2["reset_mode"], "value", full_attrs2["reset_mode"]),
                F.Word4.RESET_MODE_OFFSET,
                F.Word4.RESET_MODE_MASK,
            )
            | _p(
                _value_to_bf16_bits(full_attrs2["reset_v"]),
                F.Word4.RESET_V_OFFSET,
                F.Word4.RESET_V_MASK,
            )
            | _p(
                full_attrs2["threshold_neg_mode"],
                F.Word4.THRESHOLD_NEG_MODE_OFFSET,
                F.Word4.THRESHOLD_NEG_MODE_MASK,
            )
            | _p(
                full_attrs2["threshold_pos_mode"],
                F.Word4.THRESHOLD_POS_MODE_OFFSET,
                F.Word4.THRESHOLD_POS_MODE_MASK,
            )
            | _p(
                _value_to_bits(full_attrs2["threshold_neg"], 32),
                F.Word4.THRESHOLD_NEG_OFFSET,
                F.Word4.THRESHOLD_NEG_MASK,
            )
            | _p(
                threshold_pos_h12,
                F.Word4.THRESHOLD_POS_HIGH12_OFFSET,
                F.Word4.THRESHOLD_POS_HIGH12_MASK,
            )
        )
        return np.r_[pkg_half_neu, w3, w4].astype(FRAME_DTYPE)

    @staticmethod
    def _gen_pkg_folded_neu(
        folded_attrs1: dict[str, Any], *folded_attrs2: dict[str, Any]
    ) -> FrameArrayType:
        F = On_Cfg3_V2.Fold
        fold_skew_y_h9, fold_skew_y_l2 = bin_split(
            int(folded_attrs1["fold_skew_y"]), 2, 9
        )
        w1 = (
            _p(
                fold_skew_y_l2,
                F.Word1.FOLD_SKEW_Y_LOW2_OFFSET,
                F.Word1.FOLD_SKEW_Y_LOW2_MASK,
            )
            | _p(
                folded_attrs1["fold_axon_xy"],
                F.Word1.FOLD_AXON_XY_OFFSET,
                F.Word1.FOLD_AXON_XY_MASK,
            )
            | _p(
                folded_attrs1["fold_axon_x"],
                F.Word1.FOLD_AXON_X_OFFSET,
                F.Word1.FOLD_AXON_X_MASK,
            )
            | _p(
                folded_attrs1["fold_axon_y"],
                F.Word1.FOLD_AXON_Y_OFFSET,
                F.Word1.FOLD_AXON_Y_MASK,
            )
            | _p(
                folded_attrs1["fold_number"],
                F.Word1.FOLD_NUMBER_OFFSET,
                F.Word1.FOLD_NUMBER_MASK,
            )
        )
        w2 = (
            _p(
                folded_attrs1["fold_range_xy"],
                F.Word2.FOLD_RANGE_XY_OFFSET,
                F.Word2.FOLD_RANGE_XY_MASK,
            )
            | _p(
                folded_attrs1["fold_range_x"],
                F.Word2.FOLD_RANGE_X_OFFSET,
                F.Word2.FOLD_RANGE_X_MASK,
            )
            | _p(
                folded_attrs1["fold_range_y"],
                F.Word2.FOLD_RANGE_Y_OFFSET,
                F.Word2.FOLD_RANGE_Y_MASK,
            )
            | _p(
                folded_attrs1["fold_skew_xy"],
                F.Word2.FOLD_SKEW_XY_OFFSET,
                F.Word2.FOLD_SKEW_XY_MASK,
            )
            | _p(
                folded_attrs1["fold_skew_x"],
                F.Word2.FOLD_SKEW_X_OFFSET,
                F.Word2.FOLD_SKEW_X_MASK,
            )
            | _p(
                fold_skew_y_h9,
                F.Word2.FOLD_SKEW_Y_HIGH9_OFFSET,
                F.Word2.FOLD_SKEW_Y_HIGH9_MASK,
            )
        )

        if len(folded_attrs2) == 0:
            raise ValueError("at least one folded neuron attrs part2 is required")

        v0 = _array_to_bits([item["fold_vjt_0"] for item in folded_attrs2], 32).astype(
            FRAME_DTYPE
        )
        v1 = _array_to_bits([item["fold_vjt_1"] for item in folded_attrs2], 32).astype(
            FRAME_DTYPE
        )
        v2 = _array_to_bits([item["fold_vjt_2"] for item in folded_attrs2], 32).astype(
            FRAME_DTYPE
        )
        v3 = _array_to_bits([item["fold_vjt_3"] for item in folded_attrs2], 32).astype(
            FRAME_DTYPE
        )

        # Each folded entry is emitted as RAM[n][63:0], RAM[n][127:64].
        w3 = (
            _p(v1, F.Word3.FOLD_VJT_1_OFFSET, F.Word3.FOLD_VJT_1_MASK)
            | _p(v0, F.Word3.FOLD_VJT_0_OFFSET, F.Word3.FOLD_VJT_0_MASK)
        ).astype(FRAME_DTYPE)
        w4 = (
            _p(v3, F.Word4.FOLD_VJT_3_OFFSET, F.Word4.FOLD_VJT_3_MASK)
            | _p(v2, F.Word4.FOLD_VJT_2_OFFSET, F.Word4.FOLD_VJT_2_MASK)
        ).astype(FRAME_DTYPE)

        v = np.zeros(len(w3) * 2, dtype=FRAME_DTYPE)
        v[0::2] = w3
        v[1::2] = w4

        return np.r_[w1, w2, v].astype(FRAME_DTYPE)

    @staticmethod
    def gen_config_frame3_weight_pkg(
        weight: ArrayLike,
        csc_compress: bool | CSCAccelerateMode = False,
    ) -> FrameArrayType:
        weight_arr = np.asarray(weight)
        if weight_arr.ndim != 1:
            raise ValueError(
                f"'weight' must be a 1D array, but got ndim={weight_arr.ndim}."
            )

        if csc_compress != CSCAccelerateMode.DISABLE:
            return weight_csc_u16_pack(weight_arr)

        return weight_dense_u16_pack(weight_arr)

    @staticmethod
    def gen_config_frame4(
        pkt_offset: CoordZXYOffset,
        input_array: ArrayLike,
        pkt_ncopy: AERPacketZXYCopy = AERPacketZXYCopy(),
        start_addr: int = 0,
    ) -> FrameArrayType:
        packages = np.asarray(input_array, dtype=FRAME_DTYPE).ravel()
        return OnlineFrameGenV2.make_package(
            FH.CONFIG_TYPE4, pkt_offset, pkt_ncopy, start_addr, packages
        )

    @staticmethod
    def _gen_work_frame(
        header: FH,
        F: (
            type[On_Work1_V2]
            | type[On_Work2_V2]
            | type[On_Work3_V2]
            | type[On_Work4_V2]
        ),
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        timesteps: ArrayLike,
        axons: ArrayLike,
        target_lcn: int | LCN_EX,
        data: ArrayLike,
    ) -> FrameArrayType:
        data_parts = _normalize_online_work_data(data)
        ts = np.asarray(timesteps, dtype=FRAME_DTYPE).ravel()
        ax = np.asarray(axons, dtype=FRAME_DTYPE).ravel()

        if ax.size != ts.size:
            raise ValueError(
                f"the size of axons & timeslots are not equal, {ax.size} != {ts.size}."
            )
        if data_parts.shape[0] != ts.size:
            raise ValueError(
                f"the size of data & timeslots are not equal, {data_parts.shape[0]} != {ts.size}."
            )

        ts_width, ax_width = OnlineFrameGenV2.LCN_TO_TS_AXON_WIDTHS[
            _normalize_online_lcn(target_lcn)
        ]
        ts_ax_addr = _p(
            ((ts & _mask(ts_width)) << ax_width) | (ax & _mask(ax_width)),
            F.TIMESTEP_AXON_OFFSET,
            F.TIMESTEP_AXON_MASK,
        )

        mask = np.flatnonzero(np.any(data_parts != 0, axis=1))
        if mask.size == 0:
            return np.array([], dtype=FRAME_DTYPE)

        frame_dest = get_frame_dest_v2(header, pkt_offset, pkt_ncopy)
        data_u8 = data_parts[mask].reshape(-1).astype(FRAME_DTYPE)
        ts_ax_addr = np.repeat(ts_ax_addr[mask], data_parts.shape[1])
        return (frame_dest + ts_ax_addr + data_u8).astype(FRAME_DTYPE)

    @staticmethod
    def gen_work_frame1(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        timesteps: ArrayLike,
        axons: ArrayLike,
        target_lcn: int | LCN_EX,
        data: ArrayLike,
    ) -> FrameArrayType:
        return OnlineFrameGenV2._gen_work_frame(
            FH.WORK_TYPE1,
            On_Work1_V2,
            pkt_offset,
            pkt_ncopy,
            timesteps,
            axons,
            target_lcn,
            data,
        )

    @staticmethod
    def gen_work_frame2(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        timesteps: ArrayLike,
        axons: ArrayLike,
        target_lcn: int | LCN_EX,
        data: ArrayLike,
    ) -> FrameArrayType:
        return OnlineFrameGenV2._gen_work_frame(
            FH.WORK_TYPE2,
            On_Work2_V2,
            pkt_offset,
            pkt_ncopy,
            timesteps,
            axons,
            target_lcn,
            data,
        )

    @staticmethod
    def gen_work_frame3(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        timesteps: ArrayLike,
        axons: ArrayLike,
        target_lcn: int | LCN_EX,
        data: ArrayLike,
    ) -> FrameArrayType:
        return OnlineFrameGenV2._gen_work_frame(
            FH.WORK_TYPE3,
            On_Work3_V2,
            pkt_offset,
            pkt_ncopy,
            timesteps,
            axons,
            target_lcn,
            data,
        )

    @staticmethod
    def gen_work_frame4(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        timesteps: ArrayLike,
        axons: ArrayLike,
        target_lcn: int | LCN_EX,
        data: ArrayLike,
    ) -> FrameArrayType:
        return OnlineFrameGenV2._gen_work_frame(
            FH.WORK_TYPE4,
            On_Work4_V2,
            pkt_offset,
            pkt_ncopy,
            timesteps,
            axons,
            target_lcn,
            data,
        )

    @staticmethod
    def gen_control_frame1(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        n_timestep: int,
    ) -> FrameArrayType:
        F = On_Ctrl1_V2
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
        F = On_Ctrl3_V2
        if thread_id > F.THREAD_ID_MASK:
            raise ValueError(f"'thread_id' out of range {F.THREAD_ID_MASK})")

        return FrameV2(FH.CTRL_TYPE3, pkt_offset, pkt_ncopy, thread_id).value

    @staticmethod
    def gen_update_frame(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        ext_multicast_addr: int,
    ) -> FrameArrayType:
        F = On_Ctrl4_V2
        if ext_multicast_addr > F.EXT_MULTICAST_ADDR_MASK:
            raise ValueError(
                f"'ext_multicast_addr' out of range {F.EXT_MULTICAST_ADDR_MASK}"
            )

        return FrameV2(FH.CTRL_TYPE4, pkt_offset, pkt_ncopy, ext_multicast_addr).value

    gen_control_frame3 = gen_complete_frame
    gen_control_frame4 = gen_update_frame


# class OfflineFrameDecoderV2:
#     @staticmethod
#     def decode_voltage(frame: FrameArrayType, target_lcn: LCN_EX) -> FrameArrayType:
#         pass


def weight_dense_u16_pack(weight: np.ndarray) -> FrameArrayType:
    """Array 16-bit dense weights in RAM."""

    weight_u64 = _array_to_bf16_bits(weight).astype(FRAME_DTYPE).ravel()
    align_size = 8
    aligned_size = math.ceil(weight_u64.size / align_size) * align_size
    if (pad := aligned_size - weight_u64.size) > 0:
        weight_u64 = np.pad(weight_u64, (0, pad), constant_values=0)

    if weight_u64.size == 0:
        return np.array([], dtype=FRAME_DTYPE)

    weight_u64 = weight_u64.reshape(-1, align_size)
    shifts = 16 * np.arange(4, dtype=np.uint8)
    result = np.zeros((weight_u64.shape[0] * 2,), dtype=FRAME_DTYPE)
    result[0::2] = np.bitwise_or.reduce(
        weight_u64[:, :4] << shifts, axis=1, dtype=FRAME_DTYPE
    )
    result[1::2] = np.bitwise_or.reduce(
        weight_u64[:, 4:] << shifts, axis=1, dtype=FRAME_DTYPE
    )
    return result


def weight_csc_u16_pack(weight: np.ndarray) -> FrameArrayType:
    """Array 16-bit CSC weights in RAM."""

    weight_arr = np.asarray(weight).ravel()
    row_indices = np.flatnonzero(weight_arr != 0)
    if row_indices.size == 0:
        return np.array([], dtype=FRAME_DTYPE)

    weight_u64 = _array_to_bf16_bits(weight_arr).astype(FRAME_DTYPE).ravel()
    w_nonzero = weight_u64[row_indices]
    n_nonzero_w_per_addr = 4
    n_chunk = math.ceil(row_indices.size / n_nonzero_w_per_addr)

    if (pad := n_chunk * n_nonzero_w_per_addr - row_indices.size) > 0:
        if row_indices.size == weight_arr.size:
            raise ValueError(
                "the sparse weight cannot be aligned in groups of 4 "
                + "because there are all non-zero values."
            )

        idx_first_zero = np.where(weight_arr == 0)[0][0]
        w_nonzero = np.pad(w_nonzero, (0, pad), constant_values=0)
        row_indices = np.pad(row_indices, (0, pad), constant_values=idx_first_zero)

    w_chunks = w_nonzero.reshape(n_chunk, n_nonzero_w_per_addr)
    idx_chunks = row_indices.reshape(n_chunk, n_nonzero_w_per_addr).astype(FRAME_DTYPE)
    shifts = 16 * np.arange(n_nonzero_w_per_addr, dtype=np.uint8)

    result = np.zeros((2 * n_chunk,), dtype=FRAME_DTYPE)
    result[0::2] = np.bitwise_or.reduce(w_chunks << shifts, axis=1, dtype=FRAME_DTYPE)
    result[1::2] = np.bitwise_or.reduce(idx_chunks << shifts, axis=1, dtype=FRAME_DTYPE)
    return result


def weight_dense_pack(weight: np.ndarray, weight_width: DataWidthLE8) -> FrameArrayType:
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
    weight: np.ndarray,
    weight_width: DataWidthLE8,
    input_width: DataWidthLE8,
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

    # in csc pack, the indice stored in RAM is the bit offset of the non-zero weight,
    # which is the original index multiplied by input_width.
    row_indices = row_indices * input_width
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
    frames: FrameArrayType, weight_width: DataWidthLE8, signed: bool, original_size: int
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
    weight_width: DataWidthLE8,
    signed: bool,
    original_size: int,
) -> NDArray[np.int8 | np.uint8]:
    # #N of non-zero weight stored in a single address of RAM
    N_NONZERO_WEIGHT_PER_ADDR = {1: 7, 2: 7, 4: 6, 8: 5}
    INDICES_ADDR_OFFSET = {1: 16, 2: 16, 4: 32, 8: 48}
    n_nonzero_w_per_addr = N_NONZERO_WEIGHT_PER_ADDR[weight_width]

    if frames.size % 2 != 0:
        raise ValueError(
            f"'frames' length must be even for CSC unpack, but got {frames.size}."
        )

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
