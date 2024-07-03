import sys
from typing import Any, Literal, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    InstanceOf,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic.type_adapter import TypeAdapter
from pydantic.types import NonNegativeInt

# Use `typing_extensions.TypedDict`
from typing_extensions import TypedDict

if sys.version_info >= (3, 11):
    from typing import NotRequired
else:
    from typing_extensions import NotRequired

from .framelib.frame_defs import _mask
from .hw_defs import HwParams
from .ram_types import *

__all__ = ["NeuronDestInfo", "NeuronAttrs", "NeuronConf"]

L = Literal

# Constant of neuron destination information
TICK_RELATIVE_BIT_MAX = 8  # Unsigned
ADDR_AXON_BIT_MAX = 11  # Unsigned. Use `HwParams.ADDR_AXON_MAX` as the high limit
ADDR_CORE_X_BIT_MAX = 5  # Unsigned
ADDR_CORE_Y_BIT_MAX = 5  # Unsigned
ADDR_CORE_X_RID_BIT_MAX = 5  # Unsigned
ADDR_CORE_Y_RID_BIT_MAX = 5  # Unsigned
ADDR_CHIP_X_BIT_MAX = 5  # Unsigned
ADDR_CHIP_Y_BIT_MAX = 5  # Unsigned

TICK_RELATIVE_MAX = _mask(TICK_RELATIVE_BIT_MAX)
ADDR_AXON_MAX = HwParams.ADDR_AXON_MAX

# Constant of neuron attributes
PRN_SYN_INTGR_BIT_MAX = 1  # Unsigned
PRN_THRES_BIT_MAX = 29  # Unsigned
PRN_LEAK_V_BIT_MAX = 29  # Unsigned
RESET_V_BIT_MAX = 30  # Signed
THRES_MASK_CTRL_BIT_MAX = 5  # Unsigned
NEG_THRES_BIT_MAX = 29  # Unsigned
POS_THRES_BIT_MAX = 29  # Unsigned
LEAK_V_BIT_MAX = 30  # Signed
BIT_TRUNCATE_BIT_MAX = 5  # Unsigned
VJT_PRE_BIT_MAX = 30  # Signed

RESET_V_MAX = _mask(RESET_V_BIT_MAX - 1)
RESET_V_MIN = -(RESET_V_MAX + 1)
THRES_MASK_CTRL_MAX = _mask(THRES_MASK_CTRL_BIT_MAX)
THRES_MASK_CTRL_MIN = 0
NEG_THRES_MAX = _mask(NEG_THRES_BIT_MAX)
NEG_THRES_MIN = 0
POS_THRES_MAX = _mask(POS_THRES_BIT_MAX)
POS_THRES_MIN = 0
LEAK_V_MAX = _mask(LEAK_V_BIT_MAX - 1)  # Only for scalar
LEAK_V_MIN = -(LEAK_V_MAX + 1)  # Only for scalar
BIT_TRUNCATE_MAX = 29  # The highest bit is the sign bit
BIT_TRUNCATE_MIN = 0
VJT_PRE_MAX = _mask(VJT_PRE_BIT_MAX - 1)
VJT_PRE_MIN = -(VJT_PRE_MAX + 1)
VJT_MAX = VJT_PRE_MAX
VJT_MIN = VJT_PRE_MIN


class NeuronDestInfo(BaseModel):
    model_config = ConfigDict(
        extra="ignore", frozen=True, validate_assignment=True, strict=True
    )

    addr_chip_x: NonNegativeInt = Field(
        le=_mask(ADDR_CHIP_X_BIT_MAX), description="X coordinate of the chip."
    )

    addr_chip_y: NonNegativeInt = Field(
        le=_mask(ADDR_CHIP_Y_BIT_MAX), description="Y coordinate of the chip."
    )

    addr_core_x: NonNegativeInt = Field(
        le=_mask(ADDR_CORE_X_BIT_MAX), description="X coordinate of the core."
    )

    addr_core_y: NonNegativeInt = Field(
        le=_mask(ADDR_CORE_Y_BIT_MAX), description="Y coordinate of the core."
    )

    addr_core_x_ex: NonNegativeInt = Field(
        le=_mask(ADDR_CORE_X_RID_BIT_MAX),
        description="X replication identifier bit of the core.",
    )

    addr_core_y_ex: NonNegativeInt = Field(
        le=_mask(ADDR_CORE_Y_RID_BIT_MAX),
        description="Y replication identifier bit of the core.",
    )

    tick_relative: list[InstanceOf[NonNegativeInt]] = Field(
        description="Information of relative ticks.",
    )

    addr_axon: list[InstanceOf[NonNegativeInt]] = Field(
        description="Destination axon address."
    )

    @field_validator("tick_relative")
    @classmethod
    def _tick_relative_check(cls, v):
        if any(tr > _mask(TICK_RELATIVE_BIT_MAX) or tr < 0 for tr in v):
            # DO NOT change the type of exception `ValueError` in the validators below.
            raise ValueError("parameter 'tick relative' out of range.")

        return v

    @field_validator("addr_axon")
    @classmethod
    def _addr_axon_check(cls, v):
        if any(addr > ADDR_AXON_MAX or addr < 0 for addr in v):
            raise ValueError("parameter 'addr_axon' out of range.")

        return v

    @model_validator(mode="after")
    def _length_match_check(self):
        if len(self.tick_relative) != len(self.addr_axon):
            raise ValueError(
                "parameter 'tick relative' & 'addr_axon' must have the same "
                f"length, but {len(self.tick_relative)} != {len(self.addr_axon)}."
            )

        return self


class NeuronAttrs(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
        use_enum_values=True,
        strict=True,
    )

    reset_mode: ResetMode = Field(
        description="Reset mode of neuron.",
    )

    reset_v: int = Field(
        ge=RESET_V_MIN,
        le=RESET_V_MAX,
        description="Reset voltage of membrane potential, 30-bit signed integer.",
    )

    leak_comparison: LeakComparisonMode = Field(
        serialization_alias="leak_post",
        description="Leak after threshold comparison or before.",
    )

    threshold_mask_bits: NonNegativeInt = Field(
        le=THRES_MASK_CTRL_MAX,
        serialization_alias="threshold_mask_ctrl",
        description="X-bits mask for random threshold.",
    )

    neg_thres_mode: NegativeThresholdMode = Field(
        serialization_alias="threshold_neg_mode",
        description="Modes of negative threshold.",
    )

    neg_threshold: NonNegativeInt = Field(
        le=NEG_THRES_MAX,
        serialization_alias="threshold_neg",
        description="Negative threshold, 29-bit unsigned integer.",
    )

    pos_threshold: NonNegativeInt = Field(
        le=POS_THRES_MAX,
        serialization_alias="threshold_pos",
        description="Positive threshold, 29-bit unsigned integer.",
    )

    leak_direction: LeakDirectionMode = Field(
        serialization_alias="leak_reversal_flag",
        description="Direction of leak, forward or reversal.",
    )

    leak_integration_mode: LeakIntegrationMode = Field(
        serialization_alias="leak_det_stoch",
        description="Modes of leak integration, deterministic or stochastic.",
    )

    leak_v: Union[int, NDArray[np.int32]] = Field(
        # ge=LEAK_V_MIN,
        # le=LEAK_V_MAX,
        description="Leak voltage, 30-bit signed integer or a np.int32 array.",
    )

    synaptic_integration_mode: SynapticIntegrationMode = Field(
        serialization_alias="weight_det_stoch",
        description="Modes of synaptic integration, deterministic or stochastic.",
    )

    bit_truncation: NonNegativeInt = Field(
        le=BIT_TRUNCATE_MAX,
        serialization_alias="bit_truncate",
        description="Position of truncation, 5-bit unsigned integer.",
    )

    vjt_init: L[0] = Field(
        default=0,
        description="Initial membrane potential, 30-bit signed integer. Fixed at 0 at initialization.",
    )

    @field_serializer("leak_v", when_used="json")
    def _leak_v(self, leak_v: Union[int, NDArray[np.int32]]) -> Union[int, list[int]]:
        if isinstance(leak_v, np.ndarray):
            return leak_v.tolist()
        else:
            return leak_v


class NeuronConf(BaseModel):
    attrs: NeuronAttrs
    dest_info: NeuronDestInfo


class _NeuronAttrsDict(TypedDict):
    """Typed dictionary of `NeuronAttrs` for typing check."""

    reset_mode: int
    reset_v: int
    leak_post: int
    threshold_mask_ctrl: NonNegativeInt
    threshold_neg_mode: int
    threshold_neg: NonNegativeInt
    threshold_pos: NonNegativeInt
    leak_reversal_flag: int
    leak_det_stoch: int
    # Risky type. However, NDArray[np.int32] cannot be verified in TypedDict.
    leak_v: Union[int, Any]
    weight_det_stoch: int
    bit_truncate: NonNegativeInt
    vjt_init: NotRequired[L[0]]


class _NeuronDestInfoDict(TypedDict):
    """Typed dictionary of `NeuronDestInfo` for typing check."""

    addr_core_x: NonNegativeInt
    addr_core_y: NonNegativeInt
    addr_core_x_ex: NonNegativeInt
    addr_core_y_ex: NonNegativeInt
    addr_chip_x: NonNegativeInt
    addr_chip_y: NonNegativeInt
    tick_relative: list[InstanceOf[NonNegativeInt]]
    addr_axon: list[InstanceOf[NonNegativeInt]]


NeuronAttrsChecker = TypeAdapter(_NeuronAttrsDict)
NeuronDestInfoChecker = TypeAdapter(_NeuronDestInfoDict)
