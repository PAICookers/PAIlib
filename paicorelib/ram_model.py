from typing import List

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
from typing_extensions import TypedDict  # Use `typing_extensions.TypedDict`.
from typing_extensions import NotRequired

from .hw_defs import HwParams
from .hw_types import AxonCoord
from .ram_types import *

__all__ = ["NeuronDestInfo", "NeuronAttrs"]

TICK_RELATIVE_BIT_MAX = 8
ADDR_CORE_X_BIT_MAX = 5
ADDR_CORE_Y_BIT_MAX = 5
ADDR_CORE_X_EX_BIT_MAX = 5
ADDR_CORE_Y_EX_BIT_MAX = 5
ADDR_CHIP_X_BIT_MAX = 5
ADDR_CHIP_Y_BIT_MAX = 5


class _BasicNeuronDest(BaseModel):
    """Parameter model of RAM parameters listed in Section 2.4.2.

    NOTE: The parameters input in the model are declared in `docs/Table-of-Terms.md`.
    """

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    addr_chip_x: NonNegativeInt = Field(
        lt=(1 << ADDR_CHIP_X_BIT_MAX), description="Address X of destination chip."
    )

    addr_chip_y: NonNegativeInt = Field(
        lt=(1 << ADDR_CHIP_Y_BIT_MAX), description="Address Y of destination chip."
    )

    addr_core_x: NonNegativeInt = Field(
        lt=(1 << ADDR_CORE_X_BIT_MAX), description="Address X of destination core."
    )

    addr_core_y: NonNegativeInt = Field(
        lt=(1 << ADDR_CORE_Y_BIT_MAX), description="Address Y of destination core."
    )

    addr_core_x_ex: NonNegativeInt = Field(
        lt=(1 << ADDR_CORE_X_EX_BIT_MAX),
        description="Broadcast address X of destination core.",
    )

    addr_core_y_ex: NonNegativeInt = Field(
        lt=(1 << ADDR_CORE_Y_EX_BIT_MAX),
        description="Broadcast address Y of destination core.",
    )


class NeuronDestInfo(_BasicNeuronDest):
    tick_relative: List[InstanceOf[NonNegativeInt]] = Field(
        description="Information of relative ticks.",
    )

    addr_axon: List[InstanceOf[NonNegativeInt]] = Field(
        description="Destination axon address."
    )

    @field_validator("tick_relative")
    @classmethod
    def _tick_relative_check(cls, v):
        if any(tr >= (1 << TICK_RELATIVE_BIT_MAX) or tr < 0 for tr in v):
            # DO NOT change the type of exception `ValueError` in the validators below.
            raise ValueError("parameter 'tick relative' out of range.")

        return v

    @field_validator("addr_axon")
    @classmethod
    def _addr_axon_check(cls, v):
        if any(addr > HwParams.ADDR_AXON_MAX or addr < 0 for addr in v):
            raise ValueError("parameter 'addr_axon' out of range.")

        return v

    @model_validator(mode="after")
    def _length_match_check(self):
        if len(self.tick_relative) != len(self.addr_axon):
            raise ValueError(
                "parameter 'tick relative' & 'addr_axon' must have the same "
                f"length: {len(self.tick_relative)}, {len(self.addr_axon)}."
            )

        return self


class OutpuNeuronDestInfo(_BasicNeuronDest):
    # TODO
    start_axon_coord: AxonCoord = Field(description="Address X of destination chip.")
    end_axon_coord: AxonCoord = Field(description="Address X of destination chip.")


RESET_V_BIT_MAX = 30
THRESHOLD_MASK_CTRL_BIT_MAX = 5
NEGATIVE_THRESHOLD_VALUE_BIT_MAX = 29
POSITIVE_THRESHOLD_VALUE_BIT_MAX = 29
LEAK_V_BIT_MAX = 30
BIT_TRUNCATE_BIT_MAX = 5
VJT_PRE_BIT_MAX = 30


class NeuronAttrs(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    reset_mode: ResetMode = Field(
        description="Reset mode of neuron.",
    )

    reset_v: int = Field(
        ge=-(1 << (RESET_V_BIT_MAX - 1)),
        lt=(1 << (RESET_V_BIT_MAX - 1)),
        description="Reset value of membrane potential, 30-bit signed.",
    )

    leak_comparison: LeakComparisonMode = Field(
        serialization_alias="leak_post",
        description="Leak after threshold comparison or before.",
    )

    threshold_mask_bits: NonNegativeInt = Field(
        lt=(1 << THRESHOLD_MASK_CTRL_BIT_MAX),
        serialization_alias="threshold_mask_ctrl",
        description="X-bits mask for random threshold.",
    )

    neg_thres_mode: NegativeThresholdMode = Field(
        serialization_alias="threshold_neg_mode",
        description="Modes of negative threshold.",
    )

    neg_threshold: NonNegativeInt = Field(
        lt=(1 << NEGATIVE_THRESHOLD_VALUE_BIT_MAX),
        serialization_alias="threshold_neg",
        description="Negative threshold, 29-bit unsigned.",
    )

    pos_threshold: NonNegativeInt = Field(
        lt=(1 << POSITIVE_THRESHOLD_VALUE_BIT_MAX),
        serialization_alias="threshold_pos",
        description="Positive threshold, 29-bit unsigned.",
    )

    leak_direction: LeakDirectionMode = Field(
        serialization_alias="leak_reversal_flag",
        description="Direction of leak, forward or reversal.",
    )

    leak_integration_mode: LeakIntegrationMode = Field(
        serialization_alias="leak_det_stoch",
        description="Modes of leak integration, deterministic or stochastic.",
    )

    leak_v: int = Field(
        ge=-(1 << (LEAK_V_BIT_MAX - 1)),
        lt=(1 << (LEAK_V_BIT_MAX - 1)),
        description="Leak voltage, 30-bit signed.",
    )

    synaptic_integration_mode: SynapticIntegrationMode = Field(
        serialization_alias="weight_det_stoch",
        description="Modes of synaptic integration, deterministic or stochastic.",
    )

    bit_truncation: NonNegativeInt = Field(
        lt=(1 << BIT_TRUNCATE_BIT_MAX),
        serialization_alias="bit_truncate",
        description="Position of truncation, unsigned int, 5-bits.",
    )

    vjt_init: int = 0
    """Membrane potential of neuron at last timestep, 30-bit signed. Must be 0 at initialization."""

    @field_serializer("reset_mode")
    def _reset_mode(self, reset_mode: ResetMode) -> int:
        return reset_mode.value

    @field_serializer("leak_comparison")
    def _leak_comparison(self, leak_comparison: LeakComparisonMode) -> int:
        return leak_comparison.value

    @field_serializer("neg_thres_mode")
    def _neg_thres_mode(self, neg_thres_mode: NegativeThresholdMode) -> int:
        return neg_thres_mode.value

    @field_serializer("leak_direction")
    def _leak_direction(self, leak_direction: LeakDirectionMode) -> int:
        return leak_direction.value

    @field_serializer("leak_integration_mode")
    def _lim(self, lim: LeakIntegrationMode) -> int:
        return lim.value

    @field_serializer("synaptic_integration_mode")
    def _sim(self, sim: SynapticIntegrationMode) -> int:
        return sim.value


class _NeuronAttrsDict(TypedDict):
    """Typed dictionary of `NeuronAttrs` for typing check."""

    reset_mode: int
    reset_v: int
    leak_post: int
    threshold_mask_ctrl: int
    threshold_neg_mode: int
    threshold_neg: int
    threshold_pos: int
    leak_reversal_flag: int
    leak_det_stoch: int
    leak_v: int
    weight_det_stoch: int
    bit_truncate: int
    vjt_init: NotRequired[int]  # Reserved.


class _NeuronDestInfoDict(TypedDict):
    """Typed dictionary of `NeuronDestInfo` for typing check."""

    addr_core_x: int
    addr_core_y: int
    addr_core_x_ex: int
    addr_core_y_ex: int
    addr_chip_x: int
    addr_chip_y: int
    tick_relative: List[int]
    addr_axon: List[int]


NeuronAttrsChecker = TypeAdapter(_NeuronAttrsDict)
NeuronDestInfoChecker = TypeAdapter(_NeuronDestInfoDict)
