import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..coordinate import CoordZXYOffset, coordzxy_to_sign_magnitude
from ..core_defs import LCN_EX
from ..core_defs_v2 import CSCAccelerateMode, DataWidth
from ..core_model_v2 import OfflineCoreRegV2, OnlineCoreRegV2
from ..float_codec import (
    pack_bf16_payload_bits,
    pack_bf16_scalar_bits,
    pack_fp32_payload_bits,
    pack_fp32_scalar_bits,
)
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
from .frame_defs import OfflineWorkFrame2FormatV2 as Off_Work2_V2
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
    VOLTAGE_DTYPE,
    FrameArrayType,
    LUTActivationType,
    LUTPotentialType,
    PayloadDataType,
    VoltageDataType,
)
from .utils import TruncationWarning, _mask, bin_split, pack_field

__all__ = ["FrameGenV2", "OfflineFrameGenV2", "OnlineFrameGenV2", "WorkFrameBase"]

DataWidthLE8 = Literal[1, 2, 4, 8]
DataWidthLE8Like = DataWidth | DataWidthLE8
OfflineWorkFrameFormat = type[Off_Work1_V2 | Off_Work2_V2]
OnlineWorkFrameFormat = type[On_Work1_V2 | On_Work2_V2 | On_Work3_V2 | On_Work4_V2]
PackWorkAddr = Callable[[FrameArrayType, FrameArrayType], FrameArrayType]
N_FRAME_PER_LUT_RAM = 256
OFFLINE_WORK_TOTAL_WIDTH = 17
OFFLINE_WORK_AXON_WIDTH = 9
OFFLINE_WORK_TIMESTEP_WIDTH = OFFLINE_WORK_TOTAL_WIDTH - OFFLINE_WORK_AXON_WIDTH
ONLINE_WORK_TOTAL_WIDTH = 16
ONLINE_WORK_AXON_WIDTH = 8
ONLINE_WORK_TIMESTEP_WIDTH = ONLINE_WORK_TOTAL_WIDTH - ONLINE_WORK_AXON_WIDTH
CSC_WEIGHT_INDEX_BITS = 16
CSC_WEIGHT_INDEX_MAX = (1 << CSC_WEIGHT_INDEX_BITS) - 1

_p = pack_field


def _normalize_width_le8(width: DataWidthLE8Like) -> DataWidthLE8:
    width_bits = (1 << width.value) if isinstance(width, DataWidth) else int(width)
    if width_bits not in (1, 2, 4, 8):
        raise ValueError(f"only supports 1/2/4/8-bit widths, got {width}.")
    return width_bits


def _normalize_lcn(target_lcn: LCN_EX | int) -> int:
    """Return the integer LCN index used by V2 work-frame address packing."""
    if isinstance(target_lcn, LCN_EX):
        return target_lcn.value
    if target_lcn < LCN_EX.LCN_1X or target_lcn > LCN_EX.LCN_128X:
        raise ValueError(
            f"'target_lcn' must be in range [{LCN_EX.LCN_1X}, {LCN_EX.LCN_128X}], got {target_lcn}."
        )
    return target_lcn


def _pack_offline_work_addr(
    ts: FrameArrayType, ax: FrameArrayType, F: type[Off_Work1_V2 | Off_Work2_V2]
) -> FrameArrayType:
    """Pack offline work-frame timestamp and axon fields."""
    ts_ax_addr = (
        (ts & _mask(OFFLINE_WORK_TIMESTEP_WIDTH)) << OFFLINE_WORK_AXON_WIDTH
    ) | (ax & _mask(OFFLINE_WORK_AXON_WIDTH))
    return _p(ts_ax_addr >> 16, F.TIMESTEP_HIGH7_OFFSET, F.TIMESTEP_HIGH7_MASK) | (
        (ts_ax_addr & _mask(16)) << F.AXON_ADDR_OFFSET
    ).astype(FRAME_DTYPE)


def _pack_online_work_addr(
    ts: FrameArrayType,
    ax: FrameArrayType,
    F: type[On_Work1_V2 | On_Work2_V2 | On_Work3_V2 | On_Work4_V2],
) -> FrameArrayType:
    """Pack online work-frame timestamp and axon fields."""
    ts_ax_addr = (
        (ts & _mask(ONLINE_WORK_TIMESTEP_WIDTH)) << ONLINE_WORK_AXON_WIDTH
    ) | (ax & _mask(ONLINE_WORK_AXON_WIDTH))
    return _p(ts_ax_addr, F.TIMESTEP_AXON_OFFSET, F.TIMESTEP_AXON_MASK).astype(
        FRAME_DTYPE
    )


def _as_1d_payload_array(data: ArrayLike) -> np.ndarray:
    return np.asarray(data).ravel()


def _normalize_work_frame_tick_ax(
    tick_relatives: ArrayLike, axons: ArrayLike
) -> tuple[np.ndarray, FrameArrayType]:
    tick = np.asarray(tick_relatives).ravel()
    ax = np.asarray(axons, dtype=FRAME_DTYPE).ravel()

    if ax.size != tick.size:
        raise ValueError(
            f"the size of axons & tick_relatives are not equal, "
            f"{ax.size} != {tick.size}."
        )

    return tick, ax


def _resolve_work_frame_timestep(
    tick_relatives: np.ndarray, target_lcn: int, timestep: int, timestep_width: int
) -> FrameArrayType:
    """Resolve mapping-local ticks into the timestamp written to work frames.

    Compilation artifacts provide ``tick_relative`` values inside one LCN group.
    Runtime inference supplies the outer ``timestep``. The frame timestamp field is:

    ``cur_ts = (timestep << target_lcn) + tick_relative``
    """
    _check_work_frame_tick_relative(tick_relatives, target_lcn)
    timestep_base = _work_frame_timestep_base(target_lcn, timestep, timestep_width)
    return _resolve_work_frame_timestep_from_base(tick_relatives, timestep_base)


def _check_work_frame_tick_relative(
    tick_relatives: np.ndarray, target_lcn: int
) -> None:
    """Validate mapping-local ticks without allocating resolved timestamps."""
    if tick_relatives.size == 0:
        return

    tick_min, tick_max = tick_relatives.min(), tick_relatives.max()
    if tick_min < 0:
        raise ValueError(
            f"'tick_relatives' must be non-negative, got min {int(tick_min)}."
        )

    tick_limit = 1 << target_lcn
    if tick_max >= tick_limit:
        raise ValueError(
            f"'tick_relatives' must be in range [0, {tick_limit}), "
            f"got max {int(tick_max)}."
        )


def _resolve_work_frame_timestep_from_base(
    tick_relatives: np.ndarray, timestep_base: int
) -> FrameArrayType:
    if tick_relatives.size == 0:
        return np.array([], dtype=FRAME_DTYPE)

    tick = tick_relatives.astype(FRAME_DTYPE, copy=False)
    return tick + timestep_base


def _work_frame_timestep_base(
    target_lcn: int, timestep: int, timestep_width: int
) -> int:
    if timestep < 0:
        raise ValueError(f"'timestep' must be non-negative, got {timestep}.")

    ts_limit = 1 << timestep_width
    timestep_base = timestep << target_lcn
    if timestep_base >= ts_limit:
        raise ValueError(
            f"resolved timestep must be in range [0, {ts_limit}), "
            f"got max {timestep_base}."
        )

    return timestep_base


def _payload_to_le_bytes(arr: np.ndarray) -> PayloadDataType:
    arr = np.ascontiguousarray(arr.astype(arr.dtype.newbyteorder("<"), copy=False))
    return arr.view(PAYLOAD_DATA_DTYPE).reshape(arr.size, arr.dtype.itemsize)


@dataclass(frozen=True)
class WorkFrameBase(ABC):
    """Static work-frame state for repeated runtime payload loading.

    ``base`` contains one payload-free frame per logical input element at
    runtime ``timestep=0``. ``load()`` reuses the cached route and mapping
    arrays, then packs timestamp/axon fields only for non-zero payload entries.
    """

    base: FrameArrayType
    frame_dest: int
    tick_relatives: FrameArrayType
    axons: FrameArrayType
    target_lcn: int
    total_width: int
    axon_width: int

    @abstractmethod
    def normalize_payload(self, payload: ArrayLike) -> np.ndarray: ...

    @abstractmethod
    def pack_work_addr(
        self, ts: FrameArrayType, ax: FrameArrayType
    ) -> FrameArrayType: ...

    def load(self, payload: ArrayLike, timestep: int = 0) -> FrameArrayType:
        """Return complete work frames for ``payload`` at one runtime timestep.

        The payload is flattened and normalized using the same rules as the
        matching direct generator. Its flattened size must match ``base.size``.
        """
        normalized_payload = self.normalize_payload(payload)
        return _gen_work_frame_from_base(
            self.base,
            self.frame_dest,
            self.tick_relatives,
            self.axons,
            normalized_payload,
            self.target_lcn,
            timestep,
            self.total_width - self.axon_width,
            self.pack_work_addr,
        )


@dataclass(frozen=True)
class OfflineWorkFrameBase(WorkFrameBase):
    work_format: OfflineWorkFrameFormat

    def pack_work_addr(self, ts: FrameArrayType, ax: FrameArrayType) -> FrameArrayType:
        return _pack_offline_work_addr(ts, ax, self.work_format)


class OfflineWorkFrame1Base(OfflineWorkFrameBase):
    def normalize_payload(self, payload: ArrayLike) -> PayloadDataType:
        return np.asarray(payload, dtype=PAYLOAD_DATA_DTYPE).ravel()


class OfflineWorkFrame2Base(OfflineWorkFrameBase):
    def normalize_payload(self, payload: ArrayLike) -> VoltageDataType:
        return np.asarray(payload, dtype=VOLTAGE_DTYPE).ravel()


@dataclass(frozen=True)
class OnlineWorkFrameBase(WorkFrameBase):
    work_format: OnlineWorkFrameFormat

    def pack_work_addr(self, ts: FrameArrayType, ax: FrameArrayType) -> FrameArrayType:
        return _pack_online_work_addr(ts, ax, self.work_format)


class OnlineWorkFrame1Base(OnlineWorkFrameBase):
    def normalize_payload(self, payload: ArrayLike) -> np.ndarray:
        return _normalize_online_wf1_payload(payload)


class OnlineWorkFrame2Base(OnlineWorkFrameBase):
    def normalize_payload(self, payload: ArrayLike) -> np.ndarray:
        return _normalize_online_fp16_payload(payload, FH.WORK_TYPE2)


class OnlineWorkFrame3Base(OnlineWorkFrameBase):
    def normalize_payload(self, payload: ArrayLike) -> np.ndarray:
        return _normalize_online_wf3_payload(payload)


class OnlineWorkFrame4Base(OnlineWorkFrameBase):
    def normalize_payload(self, payload: ArrayLike) -> np.ndarray:
        return _normalize_online_fp16_payload(payload, FH.WORK_TYPE4)


def _gen_work_frame_base(
    header: FH,
    pkt_offset: CoordZXYOffset,
    pkt_ncopy: AERPacketZXYCopy,
    tick_relatives: ArrayLike,
    axons: ArrayLike,
    target_lcn: int,
    timestep_width: int,
    pack_work_addr: PackWorkAddr,
) -> FrameArrayType:
    """Build payload-free work-frame bases for ``timestep=0``.

    The returned array has one frame per logical input element. It already
    includes the route, tick-relative contribution, and axon address; runtime
    payload loading only adds the outer timestep contribution and data bytes.
    """
    _, _, _, base = _make_work_frame_base_parts(
        header,
        pkt_offset,
        pkt_ncopy,
        tick_relatives,
        axons,
        target_lcn,
        timestep_width,
        pack_work_addr,
    )
    return base


def _make_work_frame_base_parts(
    header: FH,
    pkt_offset: CoordZXYOffset,
    pkt_ncopy: AERPacketZXYCopy,
    tick_relatives: ArrayLike,
    axons: ArrayLike,
    target_lcn: int,
    timestep_width: int,
    pack_work_addr: PackWorkAddr,
) -> tuple[int, FrameArrayType, FrameArrayType, FrameArrayType]:
    tick, ax = _normalize_work_frame_tick_ax(tick_relatives, axons)
    resolved_tick = _resolve_work_frame_timestep(tick, target_lcn, 0, timestep_width)
    frame_dest = get_frame_dest_v2(header, pkt_offset, pkt_ncopy)
    frame_addr = pack_work_addr(resolved_tick, ax)
    base = (frame_dest + frame_addr).astype(FRAME_DTYPE)
    return frame_dest, tick, ax, base


def _gen_work_frame_from_base(
    base: FrameArrayType,
    frame_dest: int,
    tick_relatives: FrameArrayType,
    axons: FrameArrayType,
    payload: np.ndarray,
    target_lcn: int,
    timestep: int,
    timestep_width: int,
    pack_work_addr: PackWorkAddr,
) -> FrameArrayType:
    """Load dynamic payload and timestep into precomputed work-frame bases."""
    if payload.size != base.size:
        raise ValueError(
            f"the size of payload & work frame base are not equal, "
            f"{payload.size} != {base.size}."
        )

    if timestep == 0:
        _work_frame_timestep_base(target_lcn, timestep, timestep_width)
        mask = np.flatnonzero(payload)
        if mask.size == 0:
            return np.array([], dtype=FRAME_DTYPE)
        return _load_work_frame_payload_from_base(base[mask], payload[mask])

    _check_work_frame_tick_relative(tick_relatives, target_lcn)
    timestep_base = _work_frame_timestep_base(target_lcn, timestep, timestep_width)
    mask = np.flatnonzero(payload)
    if mask.size == 0:
        return np.array([], dtype=FRAME_DTYPE)

    resolved_timestep = _resolve_work_frame_timestep_from_base(
        tick_relatives[mask], timestep_base
    )
    return _load_work_frame_payload(
        frame_dest, resolved_timestep, axons[mask], payload[mask], pack_work_addr
    )


def _load_work_frame_payload_from_base(
    frame_base: FrameArrayType, payload: np.ndarray
) -> FrameArrayType:
    payload_parts = _payload_to_le_bytes(payload)
    frames = frame_base[:, np.newaxis] + payload_parts
    return frames.reshape(-1).astype(FRAME_DTYPE, copy=False)


def _load_work_frame_payload(
    frame_dest: int,
    resolved_timestep: FrameArrayType,
    axons: FrameArrayType,
    payload: np.ndarray,
    pack_work_addr: PackWorkAddr,
) -> FrameArrayType:
    """Fast path shared by direct generation and ``WorkFrameBase.load()``.

    Callers pass only non-zero payload entries with already-resolved timestamp
    values. Route is already scalar, so this only packs address fields before
    expanding payload bytes.
    """
    frame_base = frame_dest + pack_work_addr(resolved_timestep, axons)
    return _load_work_frame_payload_from_base(frame_base, payload)


def _gen_work_frame_bytes(
    header: FH,
    pkt_offset: CoordZXYOffset,
    pkt_ncopy: AERPacketZXYCopy,
    tick_relatives: ArrayLike,
    axons: ArrayLike,
    payload: np.ndarray,
    target_lcn: int,
    timestep: int,
    timestep_width: int,
    pack_work_addr: PackWorkAddr,
) -> FrameArrayType:
    """Generate complete work frames without precomputing a reusable base.

    The full input shape is validated before sparse filtering, but route and
    timestamp/axon packing are performed only for non-zero payload entries.
    """
    payload = _as_1d_payload_array(payload)
    tick, ax = _normalize_work_frame_tick_ax(tick_relatives, axons)

    if payload.size != tick.size:
        raise ValueError(
            f"the size of payload & tick_relatives are not equal, "
            f"{payload.size} != {tick.size}."
        )

    _check_work_frame_tick_relative(tick, target_lcn)
    timestep_base = _work_frame_timestep_base(target_lcn, timestep, timestep_width)

    mask = np.flatnonzero(payload)
    if mask.size == 0:
        return np.array([], dtype=FRAME_DTYPE)

    frame_dest = get_frame_dest_v2(header, pkt_offset, pkt_ncopy)
    resolved_timestep = _resolve_work_frame_timestep_from_base(
        tick[mask], timestep_base
    )
    return _load_work_frame_payload(
        frame_dest,
        resolved_timestep,
        ax[mask],
        payload[mask],
        pack_work_addr,
    )


def _work_frame_label(fh: FH) -> str:
    return fh.name.replace("WORK_TYPE", "WF")


def _warn_online_payload_truncation(
    fh: FH, source_dtype: np.dtype, target_dtype: np.dtype
) -> None:
    frame_name = _work_frame_label(fh)
    warnings.warn(
        f"Online {frame_name} payload dtype {source_dtype} will be truncated to "
        f"{target_dtype}.",
        TruncationWarning,
    )


def _normalize_online_wf1_payload(
    data: ArrayLike,
) -> NDArray[np.uint8] | NDArray[np.float16]:
    arr = _as_1d_payload_array(data)

    if arr.dtype == np.dtype(np.bool_):
        return arr.astype(np.uint8, copy=False)

    if arr.dtype == np.dtype(np.uint8):
        if not np.all((arr == 0) | (arr == 1)):
            raise ValueError("Online WF1 1-bit payload must contain only 0 or 1.")
        return arr

    if arr.dtype == np.dtype(np.float16):
        return arr

    if np.issubdtype(arr.dtype, np.floating):
        target_dtype = np.dtype(np.float16)
        _warn_online_payload_truncation(FH.WORK_TYPE1, arr.dtype, target_dtype)
        return arr.astype(target_dtype)

    raise TypeError(
        "Online WF1 payload must be float16 for ANN or bool/0-or-1 uint8 for SNN, "
        f"got {arr.dtype}."
    )


def _normalize_online_fp16_payload(data: ArrayLike, fh: FH) -> NDArray[np.float16]:
    arr = _as_1d_payload_array(data)
    target_dtype = np.dtype(np.float16)

    if arr.dtype == target_dtype:
        return arr

    if np.issubdtype(arr.dtype, np.floating):
        _warn_online_payload_truncation(fh, arr.dtype, target_dtype)
        return arr.astype(target_dtype)

    raise TypeError(
        f"Online {_work_frame_label(fh)} payload must be float16, got {arr.dtype}."
    )


def _normalize_online_wf3_payload(
    data: ArrayLike,
) -> NDArray[np.float16] | NDArray[np.float32]:
    arr = _as_1d_payload_array(data)
    fp16_dtype = np.dtype(np.float16)
    fp32_dtype = np.dtype(np.float32)

    if arr.dtype in (fp16_dtype, fp32_dtype):
        return arr

    if np.issubdtype(arr.dtype, np.floating):
        _warn_online_payload_truncation(FH.WORK_TYPE3, arr.dtype, fp32_dtype)
        return arr.astype(fp32_dtype)

    raise TypeError(f"Online WF3 payload must be float32 or float16, got {arr.dtype}.")


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
    """Generate V2 frames for offline cores.

    Work-frame methods take compiler-provided ``tick_relatives`` and ``axons``.
    Runtime ``timestep`` is supplied separately and resolved as
    ``(timestep << target_lcn) + tick_relative``.

    For repeated inference with fixed routing, prefer ``gen_work_frameN_static()``
    once before the loop, then call ``WorkFrameBase.load(payload, timestep)`` at
    runtime. ``gen_work_frameN_base()`` returns only the raw payload-free frame
    array for callers that do not need the object wrapper.
    """

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
        test_core_xy, test_core_x, test_core_y = coordzxy_to_sign_magnitude(
            (core_reg["test_core_xy"], core_reg["test_core_x"], core_reg["test_core_y"])
        )
        test_core_y_h2, test_core_y_l4 = bin_split(test_core_y, 4, 2)
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
            | _p(test_core_xy, F.Word1.TEST_CORE_XY_OFFSET, F.Word1.TEST_CORE_XY_MASK)
            | _p(test_core_x, F.Word1.TEST_CORE_X_OFFSET, F.Word1.TEST_CORE_X_MASK)
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
        potentials = potentials.ravel()
        activations = activations.ravel()
        if potentials.size != activations.size:
            raise ValueError(
                f"potentials and activations should have the same size, "
                f"but got {potentials.size} != {activations.size}"
            )
        if potentials.size != N_FRAME_PER_LUT_RAM:
            raise ValueError(
                f"the size of potentials and activations should be {N_FRAME_PER_LUT_RAM}, "
                f"but got {potentials.size}"
            )

        arr_pot_u32 = potentials.view(np.uint32).astype(FRAME_DTYPE)
        arr_act_u8 = activations.astype(np.uint8)
        packages = ((arr_pot_u32 << F.POTENTIAL_OFFSET) + arr_act_u8).astype(
            FRAME_DTYPE, copy=False
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
        addr_core_xy, addr_core_x, addr_core_y = coordzxy_to_sign_magnitude(
            (
                dest_info["addr_core_xy"],
                dest_info["addr_core_x"],
                dest_info["addr_core_y"],
            )
        )
        addr_copy_xy, addr_copy_x, addr_copy_y = coordzxy_to_sign_magnitude(
            (
                dest_info["addr_copy_xy"],
                dest_info["addr_copy_x"],
                dest_info["addr_copy_y"],
            )
        )
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
                addr_core_xy,
                F.Word2.ADDR_CORE_XY_OFFSET,
                F.Word2.ADDR_CORE_XY_MASK,
            )
            | _p(addr_core_x, F.Word2.ADDR_CORE_X_OFFSET, F.Word2.ADDR_CORE_X_MASK)
            | _p(addr_core_y, F.Word2.ADDR_CORE_Y_OFFSET, F.Word2.ADDR_CORE_Y_MASK)
            | _p(addr_copy_xy, F.Word2.ADDR_COPY_XY_OFFSET, F.Word2.ADDR_COPY_XY_MASK)
            | _p(addr_copy_x, F.Word2.ADDR_COPY_X_OFFSET, F.Word2.ADDR_COPY_X_MASK)
            | _p(addr_copy_y, F.Word2.ADDR_COPY_Y_OFFSET, F.Word2.ADDR_COPY_Y_MASK)
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
        return np.r_[pkg_half_neu, w3, w4]

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
        return np.r_[w1, w2, v]

    @staticmethod
    def gen_config_frame3_weight_pkg(
        weight: np.ndarray,
        weight_width: DataWidthLE8Like,
        input_width: DataWidthLE8Like,
        csc_compress: bool | CSCAccelerateMode = False,
    ) -> FrameArrayType:
        """Generate weight package for config frame type III."""
        weight = weight.ravel()
        is_compress = csc_compress != CSCAccelerateMode.DISABLE
        weight_width = _normalize_width_le8(weight_width)
        input_width = _normalize_width_le8(input_width)

        if is_compress:
            return weight_csc_pack(weight, weight_width, input_width)
        else:
            return weight_dense_pack(weight, weight_width)

    @staticmethod
    def gen_config_frame4(
        pkt_offset: CoordZXYOffset,
        input_array: FrameArrayType,
        pkt_ncopy: AERPacketZXYCopy = AERPacketZXYCopy(),
        start_addr: int = 0,
    ):
        raise NotImplementedError

    @staticmethod
    def _gen_work_frame(
        header: FH,
        F: OfflineWorkFrameFormat,
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        tick_relatives: ArrayLike,
        axons: ArrayLike,
        target_lcn: int,
        payload: np.ndarray,
        timestep: int,
    ) -> FrameArrayType:
        return _gen_work_frame_bytes(
            header,
            pkt_offset,
            pkt_ncopy,
            tick_relatives,
            axons,
            payload,
            target_lcn,
            timestep,
            OFFLINE_WORK_TIMESTEP_WIDTH,
            lambda ts, ax: _pack_offline_work_addr(ts, ax, F),
        )

    @staticmethod
    def _gen_work_frame_base(
        header: FH,
        F: OfflineWorkFrameFormat,
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        tick_relatives: ArrayLike,
        axons: ArrayLike,
        target_lcn: int,
    ) -> FrameArrayType:
        return _gen_work_frame_base(
            header,
            pkt_offset,
            pkt_ncopy,
            tick_relatives,
            axons,
            target_lcn,
            OFFLINE_WORK_TIMESTEP_WIDTH,
            lambda ts, ax: _pack_offline_work_addr(ts, ax, F),
        )

    @staticmethod
    def gen_work_frame1(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        tick_relatives: ArrayLike,
        axons: ArrayLike,
        target_lcn: LCN_EX | int,
        data: ArrayLike,
        *,
        timestep: int = 0,
    ) -> FrameArrayType:
        target_lcn = _normalize_lcn(target_lcn)
        payload = np.asarray(data, dtype=PAYLOAD_DATA_DTYPE).ravel()
        return OfflineFrameGenV2._gen_work_frame(
            FH.WORK_TYPE1,
            Off_Work1_V2,
            pkt_offset,
            pkt_ncopy,
            tick_relatives,
            axons,
            target_lcn,
            payload,
            timestep,
        )

    @staticmethod
    def gen_work_frame1_base(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        tick_relatives: ArrayLike,
        axons: ArrayLike,
        target_lcn: LCN_EX | int,
    ) -> FrameArrayType:
        target_lcn = _normalize_lcn(target_lcn)
        return OfflineFrameGenV2._gen_work_frame_base(
            FH.WORK_TYPE1,
            Off_Work1_V2,
            pkt_offset,
            pkt_ncopy,
            tick_relatives,
            axons,
            target_lcn,
        )

    @staticmethod
    def gen_work_frame1_static(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        tick_relatives: ArrayLike,
        axons: ArrayLike,
        target_lcn: LCN_EX | int,
    ) -> OfflineWorkFrame1Base:
        target_lcn = _normalize_lcn(target_lcn)
        F = Off_Work1_V2
        frame_dest, tick, ax, base = _make_work_frame_base_parts(
            FH.WORK_TYPE1,
            pkt_offset,
            pkt_ncopy,
            tick_relatives,
            axons,
            target_lcn,
            OFFLINE_WORK_TIMESTEP_WIDTH,
            lambda ts, ax: _pack_offline_work_addr(ts, ax, F),
        )
        return OfflineWorkFrame1Base(
            base,
            frame_dest,
            tick,
            ax,
            target_lcn,
            OFFLINE_WORK_TOTAL_WIDTH,
            OFFLINE_WORK_AXON_WIDTH,
            F,
        )

    @staticmethod
    def gen_work_frame2(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        tick_relatives: ArrayLike,
        axons: ArrayLike,
        target_lcn: LCN_EX | int,
        voltage: ArrayLike,
        *,
        timestep: int = 0,
    ) -> FrameArrayType:
        target_lcn = _normalize_lcn(target_lcn)
        voltage = np.asarray(voltage, dtype=VOLTAGE_DTYPE).ravel()
        return OfflineFrameGenV2._gen_work_frame(
            FH.WORK_TYPE2,
            Off_Work2_V2,
            pkt_offset,
            pkt_ncopy,
            tick_relatives,
            axons,
            target_lcn,
            voltage,
            timestep,
        )

    @staticmethod
    def gen_work_frame2_base(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        tick_relatives: ArrayLike,
        axons: ArrayLike,
        target_lcn: LCN_EX | int,
    ) -> FrameArrayType:
        target_lcn = _normalize_lcn(target_lcn)
        return OfflineFrameGenV2._gen_work_frame_base(
            FH.WORK_TYPE2,
            Off_Work2_V2,
            pkt_offset,
            pkt_ncopy,
            tick_relatives,
            axons,
            target_lcn,
        )

    @staticmethod
    def gen_work_frame2_static(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        tick_relatives: ArrayLike,
        axons: ArrayLike,
        target_lcn: LCN_EX | int,
    ) -> OfflineWorkFrame2Base:
        target_lcn = _normalize_lcn(target_lcn)
        F = Off_Work2_V2
        frame_dest, tick, ax, base = _make_work_frame_base_parts(
            FH.WORK_TYPE2,
            pkt_offset,
            pkt_ncopy,
            tick_relatives,
            axons,
            target_lcn,
            OFFLINE_WORK_TIMESTEP_WIDTH,
            lambda ts, ax: _pack_offline_work_addr(ts, ax, F),
        )
        return OfflineWorkFrame2Base(
            base,
            frame_dest,
            tick,
            ax,
            target_lcn,
            OFFLINE_WORK_TOTAL_WIDTH,
            OFFLINE_WORK_AXON_WIDTH,
            F,
        )

    @staticmethod
    def gen_control_frame1(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy = AERPacketZXYCopy(),
        n_timestep: int = 1,
    ) -> FrameArrayType:
        F = Off_Ctrl1_V2
        if n_timestep > F.N_TIMESTEP_MASK:
            raise ValueError(f"'overflow' out of range {F.N_TIMESTEP_MASK}")

        return FrameV2(FH.CTRL_TYPE1, pkt_offset, pkt_ncopy, n_timestep).value

    @staticmethod
    def gen_control_frame2(
        pkt_offset: CoordZXYOffset, pkt_ncopy: AERPacketZXYCopy = AERPacketZXYCopy()
    ) -> FrameArrayType:
        return FrameV2(FH.CTRL_TYPE2, pkt_offset, pkt_ncopy, 0).value

    @staticmethod
    def gen_control_frame3(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy = AERPacketZXYCopy(),
        thread_id: int = 0,
    ) -> FrameArrayType:
        F = Off_Ctrl3_V2
        if thread_id > F.THREAD_ID_MASK:
            raise ValueError(f"'thread_id' out of range {F.THREAD_ID_MASK})")

        return FrameV2(FH.CTRL_TYPE3, pkt_offset, pkt_ncopy, thread_id).value


class OnlineFrameGenV2(FrameGenV2):
    """Generate V2 frames for online cores.

    Work-frame timestamp semantics and static-base usage match
    ``OfflineFrameGenV2``; online work-frame variants differ only in frame
    format, payload dtype rules, and timestamp/axon field width.
    """

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
        scale_out_bits = pack_bf16_scalar_bits(core_reg["scale_out"])
        scale_out_h15, scale_out_l1 = bin_split(scale_out_bits, 1, 15)
        update_core_xy, update_core_x, update_core_y = coordzxy_to_sign_magnitude(
            (
                core_reg["update_core_xy"],
                core_reg["update_core_x"],
                core_reg["update_core_y"],
            )
        )
        test_core_xy, test_core_x, test_core_y = coordzxy_to_sign_magnitude(
            (
                core_reg["test_core_xy"],
                core_reg["test_core_x"],
                core_reg["test_core_y"],
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
            | _p(core_reg["lcn_at"], F.Word1.LCN_AT_OFFSET, F.Word1.LCN_AT_MASK)
            | _p(core_reg["lcn_mp"], F.Word1.LCN_MP_OFFSET, F.Word1.LCN_MP_MASK)
            | _p(core_reg["lcn_lg"], F.Word1.LCN_LG_OFFSET, F.Word1.LCN_LG_MASK)
            | _p(
                core_reg["target_lcn_at"],
                F.Word1.TARGET_LCN_AT_OFFSET,
                F.Word1.TARGET_LCN_AT_MASK,
            )
            | _p(
                core_reg["target_lcn_mp"],
                F.Word1.TARGET_LCN_MP_OFFSET,
                F.Word1.TARGET_LCN_MP_MASK,
            )
            | _p(
                core_reg["target_lcn_lg"],
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
                pack_bf16_scalar_bits(core_reg["scale_in"]),
                F.Word2.SCALE_IN_OFFSET,
                F.Word2.SCALE_IN_MASK,
            )
            | _p(
                pack_bf16_scalar_bits(core_reg["bias_in"]),
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
            _p(scale_out_l1, F.Word3.SCALE_OUT_LOW1_OFFSET, F.Word3.SCALE_OUT_LOW1_MASK)
            | _p(
                pack_bf16_scalar_bits(core_reg["bias_out"]),
                F.Word3.BIAS_OUT_OFFSET,
                F.Word3.BIAS_OUT_MASK,
            )
            | _p(
                pack_bf16_scalar_bits(core_reg["learning_rate"]),
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
            | _p(test_core_x, F.Word3.TEST_CORE_X_OFFSET, F.Word3.TEST_CORE_X_MASK)
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
        potentials: np.ndarray,
        activations: np.ndarray,
        pkt_ncopy: AERPacketZXYCopy = AERPacketZXYCopy(),
        start_addr: int = 0,
    ) -> FrameArrayType:
        F = On_Cfg2_V2
        arr_pot_u32 = pack_fp32_payload_bits(potentials).ravel().astype(FRAME_DTYPE)
        arr_act_u16 = pack_bf16_payload_bits(activations).ravel()

        if arr_pot_u32.size != arr_act_u16.size:
            raise ValueError(
                f"potentials & activations should have the same size, "
                f"but got {arr_pot_u32.size} != {arr_act_u16.size}"
            )
        if arr_pot_u32.size != N_FRAME_PER_LUT_RAM:
            raise ValueError(
                f"the size of potentials & activations should be {N_FRAME_PER_LUT_RAM}, "
                f"but got {arr_pot_u32.size}"
            )

        packages = ((arr_pot_u32 << F.POTENTIAL_OFFSET) + arr_act_u16).astype(
            FRAME_DTYPE, copy=False
        )
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
                dest_info["addr_core_xy"],
                dest_info["addr_core_x"],
                dest_info["addr_core_y"],
            )
        )
        addr_copy_xy, addr_copy_x, addr_copy_y = coordzxy_to_sign_magnitude(
            (
                dest_info["addr_copy_xy"],
                dest_info["addr_copy_x"],
                dest_info["addr_copy_y"],
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
                pack_fp32_scalar_bits(half_attrs["vjt"]),
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
                dest_info["addr_axon"], F.Word2.ADDR_AXON_OFFSET, F.Word2.ADDR_AXON_MASK
            )
            | _p(addr_core_xy, F.Word2.ADDR_CORE_XY_OFFSET, F.Word2.ADDR_CORE_XY_MASK)
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
            | _p(addr_copy_x, F.Word2.ADDR_COPY_X_OFFSET, F.Word2.ADDR_COPY_X_MASK)
            | _p(addr_copy_y, F.Word2.ADDR_COPY_Y_OFFSET, F.Word2.ADDR_COPY_Y_MASK)
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
            pack_fp32_scalar_bits(full_attrs2["threshold_pos"]), 20, 12
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
                full_attrs2["leak_tau"], F.Word3.LEAK_TAU_OFFSET, F.Word3.LEAK_TAU_MASK
            )
            | _p(
                pack_bf16_scalar_bits(full_attrs2["vjt_initial"]),
                F.Word3.VJT_INITIAL_OFFSET,
                F.Word3.VJT_INITIAL_MASK,
            )
            | _p(
                full_attrs2["weight_compress"],
                F.Word3.WEIGHT_COMPRESS_OFFSET,
                F.Word3.WEIGHT_COMPRESS_MASK,
            )
            | _p(
                pack_bf16_scalar_bits(full_attrs2["leak_v"]),
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
                pack_bf16_scalar_bits(full_attrs2["reset_v"]),
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
                pack_fp32_scalar_bits(full_attrs2["threshold_neg"]),
                F.Word4.THRESHOLD_NEG_OFFSET,
                F.Word4.THRESHOLD_NEG_MASK,
            )
            | _p(
                threshold_pos_h12,
                F.Word4.THRESHOLD_POS_HIGH12_OFFSET,
                F.Word4.THRESHOLD_POS_HIGH12_MASK,
            )
        )
        return np.r_[pkg_half_neu, w3, w4]

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

        v0 = pack_fp32_payload_bits(
            [item["fold_vjt_0"] for item in folded_attrs2]
        ).astype(FRAME_DTYPE)
        v1 = pack_fp32_payload_bits(
            [item["fold_vjt_1"] for item in folded_attrs2]
        ).astype(FRAME_DTYPE)
        v2 = pack_fp32_payload_bits(
            [item["fold_vjt_2"] for item in folded_attrs2]
        ).astype(FRAME_DTYPE)
        v3 = pack_fp32_payload_bits(
            [item["fold_vjt_3"] for item in folded_attrs2]
        ).astype(FRAME_DTYPE)

        # Each folded entry is emitted as RAM[n][63:0], RAM[n][127:64].
        w3 = _p(v1, F.Word3.FOLD_VJT_1_OFFSET, F.Word3.FOLD_VJT_1_MASK) | _p(
            v0, F.Word3.FOLD_VJT_0_OFFSET, F.Word3.FOLD_VJT_0_MASK
        )
        w4 = _p(v3, F.Word4.FOLD_VJT_3_OFFSET, F.Word4.FOLD_VJT_3_MASK) | _p(
            v2, F.Word4.FOLD_VJT_2_OFFSET, F.Word4.FOLD_VJT_2_MASK
        )
        v = np.zeros(len(w3) * 2, dtype=FRAME_DTYPE)
        v[0::2] = w3
        v[1::2] = w4
        return np.r_[w1, w2, v]

    @staticmethod
    def gen_config_frame3_weight_pkg(
        weight: np.ndarray,
        csc_compress: bool | CSCAccelerateMode = False,
    ) -> FrameArrayType:
        weight = weight.ravel()
        is_compress = csc_compress != CSCAccelerateMode.DISABLE
        if is_compress:
            return online_weight_csc_pack(weight)
        else:
            return online_weight_dense_pack(weight)

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
        F: OnlineWorkFrameFormat,
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        tick_relatives: ArrayLike,
        axons: ArrayLike,
        target_lcn: int,
        payload: np.ndarray,
        timestep: int,
    ) -> FrameArrayType:
        return _gen_work_frame_bytes(
            header,
            pkt_offset,
            pkt_ncopy,
            tick_relatives,
            axons,
            payload,
            target_lcn,
            timestep,
            ONLINE_WORK_TIMESTEP_WIDTH,
            lambda ts, ax: _pack_online_work_addr(ts, ax, F),
        )

    @staticmethod
    def _gen_work_frame_base(
        header: FH,
        F: OnlineWorkFrameFormat,
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        tick_relatives: ArrayLike,
        axons: ArrayLike,
        target_lcn: int,
    ) -> FrameArrayType:
        return _gen_work_frame_base(
            header,
            pkt_offset,
            pkt_ncopy,
            tick_relatives,
            axons,
            target_lcn,
            ONLINE_WORK_TIMESTEP_WIDTH,
            lambda ts, ax: _pack_online_work_addr(ts, ax, F),
        )

    @staticmethod
    def gen_work_frame1(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        tick_relatives: ArrayLike,
        axons: ArrayLike,
        target_lcn: LCN_EX | int,
        data: ArrayLike,
        *,
        timestep: int = 0,
    ) -> FrameArrayType:
        target_lcn = _normalize_lcn(target_lcn)
        v_parts = _normalize_online_wf1_payload(data)
        return OnlineFrameGenV2._gen_work_frame(
            FH.WORK_TYPE1,
            On_Work1_V2,
            pkt_offset,
            pkt_ncopy,
            tick_relatives,
            axons,
            target_lcn,
            v_parts,
            timestep,
        )

    @staticmethod
    def gen_work_frame1_base(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        tick_relatives: ArrayLike,
        axons: ArrayLike,
        target_lcn: LCN_EX | int,
    ) -> FrameArrayType:
        target_lcn = _normalize_lcn(target_lcn)
        return OnlineFrameGenV2._gen_work_frame_base(
            FH.WORK_TYPE1,
            On_Work1_V2,
            pkt_offset,
            pkt_ncopy,
            tick_relatives,
            axons,
            target_lcn,
        )

    @staticmethod
    def gen_work_frame1_static(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        tick_relatives: ArrayLike,
        axons: ArrayLike,
        target_lcn: LCN_EX | int,
    ) -> OnlineWorkFrame1Base:
        target_lcn = _normalize_lcn(target_lcn)
        F = On_Work1_V2
        frame_dest, tick, ax, base = _make_work_frame_base_parts(
            FH.WORK_TYPE1,
            pkt_offset,
            pkt_ncopy,
            tick_relatives,
            axons,
            target_lcn,
            ONLINE_WORK_TIMESTEP_WIDTH,
            lambda ts, ax: _pack_online_work_addr(ts, ax, F),
        )
        return OnlineWorkFrame1Base(
            base,
            frame_dest,
            tick,
            ax,
            target_lcn,
            ONLINE_WORK_TOTAL_WIDTH,
            ONLINE_WORK_AXON_WIDTH,
            F,
        )

    @staticmethod
    def gen_work_frame2(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        tick_relatives: ArrayLike,
        axons: ArrayLike,
        target_lcn: LCN_EX | int,
        gradient: ArrayLike,
        *,
        timestep: int = 0,
    ) -> FrameArrayType:
        target_lcn = _normalize_lcn(target_lcn)
        v_parts = _normalize_online_fp16_payload(gradient, FH.WORK_TYPE2)
        return OnlineFrameGenV2._gen_work_frame(
            FH.WORK_TYPE2,
            On_Work2_V2,
            pkt_offset,
            pkt_ncopy,
            tick_relatives,
            axons,
            target_lcn,
            v_parts,
            timestep,
        )

    @staticmethod
    def gen_work_frame2_base(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        tick_relatives: ArrayLike,
        axons: ArrayLike,
        target_lcn: LCN_EX | int,
    ) -> FrameArrayType:
        target_lcn = _normalize_lcn(target_lcn)
        return OnlineFrameGenV2._gen_work_frame_base(
            FH.WORK_TYPE2,
            On_Work2_V2,
            pkt_offset,
            pkt_ncopy,
            tick_relatives,
            axons,
            target_lcn,
        )

    @staticmethod
    def gen_work_frame2_static(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        tick_relatives: ArrayLike,
        axons: ArrayLike,
        target_lcn: LCN_EX | int,
    ) -> OnlineWorkFrame2Base:
        target_lcn = _normalize_lcn(target_lcn)
        F = On_Work2_V2
        frame_dest, tick, ax, base = _make_work_frame_base_parts(
            FH.WORK_TYPE2,
            pkt_offset,
            pkt_ncopy,
            tick_relatives,
            axons,
            target_lcn,
            ONLINE_WORK_TIMESTEP_WIDTH,
            lambda ts, ax: _pack_online_work_addr(ts, ax, F),
        )
        return OnlineWorkFrame2Base(
            base,
            frame_dest,
            tick,
            ax,
            target_lcn,
            ONLINE_WORK_TOTAL_WIDTH,
            ONLINE_WORK_AXON_WIDTH,
            F,
        )

    @staticmethod
    def gen_work_frame3(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        tick_relatives: ArrayLike,
        axons: ArrayLike,
        target_lcn: LCN_EX | int,
        voltage: ArrayLike,
        *,
        timestep: int = 0,
    ) -> FrameArrayType:
        target_lcn = _normalize_lcn(target_lcn)
        v_parts = _normalize_online_wf3_payload(voltage)
        return OnlineFrameGenV2._gen_work_frame(
            FH.WORK_TYPE3,
            On_Work3_V2,
            pkt_offset,
            pkt_ncopy,
            tick_relatives,
            axons,
            target_lcn,
            v_parts,
            timestep,
        )

    @staticmethod
    def gen_work_frame3_base(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        tick_relatives: ArrayLike,
        axons: ArrayLike,
        target_lcn: LCN_EX | int,
    ) -> FrameArrayType:
        target_lcn = _normalize_lcn(target_lcn)
        return OnlineFrameGenV2._gen_work_frame_base(
            FH.WORK_TYPE3,
            On_Work3_V2,
            pkt_offset,
            pkt_ncopy,
            tick_relatives,
            axons,
            target_lcn,
        )

    @staticmethod
    def gen_work_frame3_static(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        tick_relatives: ArrayLike,
        axons: ArrayLike,
        target_lcn: LCN_EX | int,
    ) -> OnlineWorkFrame3Base:
        target_lcn = _normalize_lcn(target_lcn)
        F = On_Work3_V2
        frame_dest, tick, ax, base = _make_work_frame_base_parts(
            FH.WORK_TYPE3,
            pkt_offset,
            pkt_ncopy,
            tick_relatives,
            axons,
            target_lcn,
            ONLINE_WORK_TIMESTEP_WIDTH,
            lambda ts, ax: _pack_online_work_addr(ts, ax, F),
        )
        return OnlineWorkFrame3Base(
            base,
            frame_dest,
            tick,
            ax,
            target_lcn,
            ONLINE_WORK_TOTAL_WIDTH,
            ONLINE_WORK_AXON_WIDTH,
            F,
        )

    @staticmethod
    def gen_work_frame4(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        tick_relatives: ArrayLike,
        axons: ArrayLike,
        target_lcn: LCN_EX | int,
        voltage_gradient: ArrayLike,
        *,
        timestep: int = 0,
    ) -> FrameArrayType:
        target_lcn = _normalize_lcn(target_lcn)
        v_parts = _normalize_online_fp16_payload(voltage_gradient, FH.WORK_TYPE4)
        return OnlineFrameGenV2._gen_work_frame(
            FH.WORK_TYPE4,
            On_Work4_V2,
            pkt_offset,
            pkt_ncopy,
            tick_relatives,
            axons,
            target_lcn,
            v_parts,
            timestep,
        )

    @staticmethod
    def gen_work_frame4_base(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        tick_relatives: ArrayLike,
        axons: ArrayLike,
        target_lcn: LCN_EX | int,
    ) -> FrameArrayType:
        target_lcn = _normalize_lcn(target_lcn)
        return OnlineFrameGenV2._gen_work_frame_base(
            FH.WORK_TYPE4,
            On_Work4_V2,
            pkt_offset,
            pkt_ncopy,
            tick_relatives,
            axons,
            target_lcn,
        )

    @staticmethod
    def gen_work_frame4_static(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy,
        tick_relatives: ArrayLike,
        axons: ArrayLike,
        target_lcn: LCN_EX | int,
    ) -> OnlineWorkFrame4Base:
        target_lcn = _normalize_lcn(target_lcn)
        F = On_Work4_V2
        frame_dest, tick, ax, base = _make_work_frame_base_parts(
            FH.WORK_TYPE4,
            pkt_offset,
            pkt_ncopy,
            tick_relatives,
            axons,
            target_lcn,
            ONLINE_WORK_TIMESTEP_WIDTH,
            lambda ts, ax: _pack_online_work_addr(ts, ax, F),
        )
        return OnlineWorkFrame4Base(
            base,
            frame_dest,
            tick,
            ax,
            target_lcn,
            ONLINE_WORK_TOTAL_WIDTH,
            ONLINE_WORK_AXON_WIDTH,
            F,
        )

    @staticmethod
    def gen_control_frame1(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy = AERPacketZXYCopy(),
        n_timestep: int = 1,
    ) -> FrameArrayType:
        F = On_Ctrl1_V2
        if n_timestep > F.N_TIMESTEP_MASK:
            raise ValueError(f"'overflow' out of range {F.N_TIMESTEP_MASK}")

        return FrameV2(FH.CTRL_TYPE1, pkt_offset, pkt_ncopy, n_timestep).value

    @staticmethod
    def gen_control_frame2(
        pkt_offset: CoordZXYOffset, pkt_ncopy: AERPacketZXYCopy = AERPacketZXYCopy()
    ) -> FrameArrayType:
        return FrameV2(FH.CTRL_TYPE2, pkt_offset, pkt_ncopy, 0).value

    @staticmethod
    def gen_control_frame3(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy = AERPacketZXYCopy(),
        thread_id: int = 0,
    ) -> FrameArrayType:
        F = On_Ctrl3_V2
        if thread_id > F.THREAD_ID_MASK:
            raise ValueError(f"'thread_id' out of range {F.THREAD_ID_MASK})")

        return FrameV2(FH.CTRL_TYPE3, pkt_offset, pkt_ncopy, thread_id).value

    @staticmethod
    def gen_control_frame4(
        pkt_offset: CoordZXYOffset,
        pkt_ncopy: AERPacketZXYCopy = AERPacketZXYCopy(),
        ext_pkt_ncopy: AERPacketZXYCopy = AERPacketZXYCopy(),
    ) -> FrameArrayType:
        F = On_Ctrl4_V2
        ext_xy, ext_x, ext_y = ext_pkt_ncopy.to_sign_magnitude()
        ext_addr = (
            _p(ext_xy, F.EXT_COPY_XY_ADDR_OFFSET, F.EXT_COPY_XY_ADDR_MASK)
            | _p(ext_x, F.EXT_COPY_X_ADDR_OFFSET, F.EXT_COPY_X_ADDR_MASK)
            | _p(ext_y, F.EXT_COPY_Y_ADDR_OFFSET, F.EXT_COPY_Y_ADDR_MASK)
        )
        return FrameV2(FH.CTRL_TYPE4, pkt_offset, pkt_ncopy, ext_addr).value


def _pad_1d_to_multiple(arr: np.ndarray, multiple: int) -> np.ndarray:
    pad = (-arr.size) % multiple
    if pad > 0:
        arr = np.pad(arr, (0, pad), constant_values=0)
    return arr


def _pack_u16_groups_to_u64(values: np.ndarray, group_size: int) -> FrameArrayType:
    values = np.ascontiguousarray(values, dtype=np.uint16).ravel()
    values = _pad_1d_to_multiple(values, group_size)
    return values.view(FRAME_DTYPE)


@overload
def _pack_unsigned_groups(
    values: np.ndarray, bit_width: int, group_size: int, *, dtype: type[np.uint8]
) -> NDArray[np.uint8]: ...


@overload
def _pack_unsigned_groups(
    values: np.ndarray, bit_width: int, group_size: int, *, dtype: type[np.uint64]
) -> FrameArrayType: ...


def _pack_unsigned_groups(
    values: np.ndarray,
    bit_width: int,
    group_size: int,
    *,
    dtype: type[np.uint8] | type[np.uint64],
) -> NDArray[np.uint8] | FrameArrayType:
    if dtype not in (np.uint8, FRAME_DTYPE):
        raise TypeError(f"'dtype' must be np.uint8 or FRAME_DTYPE, got {dtype}.")

    values = np.asarray(values, dtype=dtype).reshape(-1, group_size)
    mask = np.asarray(_mask(bit_width), dtype=dtype)
    shifts = bit_width * np.arange(group_size, dtype=dtype)
    return np.bitwise_or.reduce((values & mask) << shifts, axis=1, dtype=dtype)


def _align_sparse_payload_and_indices(
    payload: np.ndarray, row_indices: np.ndarray, group_size: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Pad one-dimensional CSC payload/index lanes to complete records.

    CSC padding lanes are explicit zero weights. Their stored row index repeats
    the last real non-zero row index, which preserves nondecreasing index order
    inside and across emitted records while keeping unpack helpers able to
    ignore padding by checking the zero payload.
    """
    pad = (-row_indices.size) % group_size
    n_chunk = (row_indices.size + pad) // group_size
    if pad == 0:
        return payload, row_indices, n_chunk

    aligned_size = row_indices.size + pad
    aligned_payload = np.zeros(aligned_size, dtype=payload.dtype)
    aligned_payload[: payload.size] = payload

    aligned_indices = np.empty(aligned_size, dtype=row_indices.dtype)
    aligned_indices[: row_indices.size] = row_indices
    aligned_indices[row_indices.size :] = row_indices[-1]
    return aligned_payload, aligned_indices, n_chunk


def weight_dense_pack(weight: np.ndarray, weight_width: DataWidthLE8) -> FrameArrayType:
    """Array uncompressed weights in RAM.

    weight[127:0] -> RAM[0][ 63: 0]
                     RAM[0][127:64]
    128/64/32/16 1/2/4/8-bit weights are stored in one address of RAM, little endian.
    """
    weight = np.asarray(weight, dtype=np.uint8).ravel()
    align_size = 128 // weight_width
    weight = _pad_1d_to_multiple(weight, align_size)

    if weight_width == 8:
        return weight.view(FRAME_DTYPE)

    weights_per_byte = 8 // weight_width
    # Reduce to u8 (N,) and then view 8*u8 as u64.
    reduced = _pack_unsigned_groups(
        weight, weight_width, weights_per_byte, dtype=np.uint8
    )
    return reduced.view(FRAME_DTYPE)


def weight_csc_pack(
    weight: np.ndarray, weight_width: DataWidthLE8, input_width: DataWidthLE8
) -> FrameArrayType:
    """Arrange compressed weights according to CSC format.

    weight[127:0]: col_indices  +   weight
    1-bit           [127:16]        [ 6:0] stored 7 non-zero 1-bit weights
    2-bit           [127:16]        [13:0] stored 7 non-zero 2-bit weights
    4-bit           [127:32]        [23:0] stored 6 non-zero 4-bit weights
    8-bit           [127:48]        [39:0] stored 5 non-zero 8-bit weights

    NOTE: Non-zero values are aligned in groups of 7/7/6/5. Padding lanes store
        zero payloads, and their indices repeat the last real non-zero index to
        preserve monotonically nondecreasing CSC indices.
    """
    weight = np.asarray(weight, dtype=np.uint8).ravel()
    # #N of non-zero weight stored in a single address of RAM
    N_NONZERO_WEIGHT_PER_ADDR = {1: 7, 2: 7, 4: 6, 8: 5}
    INDICES_ADDR_OFFSET = {1: 16, 2: 16, 4: 32, 8: 48}
    n_nonzero_w_per_addr = N_NONZERO_WEIGHT_PER_ADDR[weight_width]

    row_indices = np.flatnonzero(weight)  # idx of non-zero weight
    if row_indices.size == 0:
        return np.array([], dtype=FRAME_DTYPE)

    w_payload, row_indices, n_chunk = _align_sparse_payload_and_indices(
        weight[row_indices], row_indices, n_nonzero_w_per_addr
    )

    # In csc pack, the indice stored in RAM is the bit offset of the non-zero weight,
    # which is the original index multiplied by input_width.
    row_indices *= input_width
    # Hardware CSC `weight_indice` fields are 16-bit values.
    if row_indices.max() > CSC_WEIGHT_INDEX_MAX:
        raise ValueError(
            "CSC weight indices are stored in 16-bit fields; "
            f"got max bit index {row_indices.max()}."
        )

    w_reduced_chunk = _pack_unsigned_groups(
        w_payload, weight_width, n_nonzero_w_per_addr, dtype=FRAME_DTYPE
    )

    n_idx_at_high = 4  # 4 indices will placed at high [127:64]
    n_idx_at_low = n_nonzero_w_per_addr - n_idx_at_high
    idx_chunks = row_indices.reshape(n_chunk, n_nonzero_w_per_addr).astype(np.uint16)
    idx_reduced_chunk_h = _pack_unsigned_groups(
        idx_chunks[:, n_idx_at_low:], 16, n_idx_at_high, dtype=FRAME_DTYPE
    )
    idx_reduced_chunk_l = _pack_unsigned_groups(
        idx_chunks[:, :n_idx_at_low], 16, n_idx_at_low, dtype=FRAME_DTYPE
    )
    idx_reduced_chunk_l <<= INDICES_ADDR_OFFSET[weight_width]

    result = np.zeros((2 * n_chunk,), dtype=FRAME_DTYPE)
    # Little endian
    result[0::2] = idx_reduced_chunk_l + w_reduced_chunk
    result[1::2] = idx_reduced_chunk_h
    return result


def online_weight_dense_pack(weight: np.ndarray) -> FrameArrayType:
    """Pack dense online weights as 8 BF16 values per 128-bit RAM record."""
    return _pack_u16_groups_to_u64(pack_bf16_payload_bits(weight).ravel(), 8)


def online_weight_csc_pack(weight: np.ndarray) -> FrameArrayType:
    """Pack sparse online weights as CSC RAM records with BF16 payloads.

    Each 128-bit record stores 4 non-zero BF16 weights in ``[63:0]`` and the
    corresponding 16-bit row indices in ``[127:64]``. The result is returned as
    an ``(n,)`` ``u64`` array, so each record becomes two adjacent 64-bit words.
    """
    weight = weight.ravel()
    N_NONZERO_W_PER_ADDR = 4

    row_indices = np.flatnonzero(weight)
    if row_indices.size == 0:
        return np.array([], dtype=FRAME_DTYPE)

    # Keep only the non-zero BF16 payloads; CSC stores weights and indices in
    # parallel 4-lane groups.
    w_payload, row_indices, n_chunk = _align_sparse_payload_and_indices(
        pack_bf16_payload_bits(weight).ravel()[row_indices],
        row_indices,
        N_NONZERO_W_PER_ADDR,
    )

    result = np.zeros((2 * n_chunk,), dtype=FRAME_DTYPE)
    # A single 128-bit CSC record is emitted as two adjacent u64 words:
    # low 64 bits for 4 BF16 weights, high 64 bits for 4 uint16 row indices.
    result[0::2] = _pack_u16_groups_to_u64(w_payload, N_NONZERO_W_PER_ADDR)
    result[1::2] = _pack_u16_groups_to_u64(row_indices, N_NONZERO_W_PER_ADDR)
    return result
