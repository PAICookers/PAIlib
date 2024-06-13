from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator
from pydantic.type_adapter import TypeAdapter
from pydantic.types import NonNegativeInt

# Use `typing_extensions.TypedDict`.
from typing_extensions import TypedDict

from .coordinate import Coord
from .framelib.frame_defs import _mask
from .hw_defs import HwParams
from .reg_types import *

__all__ = ["CoreParams", "ParamsReg"]

L = Literal

NUM_DENDRITE_BIT_MAX = 13  # Unsigned
TICK_WAIT_START_BIT_MAX = 15  # Unsigned
TICK_WAIT_END_BIT_MAX = 15  # Unsigned

# Use `HwParams.N_DENDRITE_MAX_ANN` as the high limit
NUM_DENDRITE_MAX = HwParams.N_DENDRITE_MAX_ANN
TICK_WAIT_START_MAX = _mask(TICK_WAIT_START_BIT_MAX)
TICK_WAIT_END_MAX = _mask(TICK_WAIT_END_BIT_MAX)

NUM_DENDRITE_OUT_OF_RANGE_TEXT = (
    "param 'num_dendrite' out of range. When input width is 8-bit in {0} mode, "
    + "the number of dendrites should be no more than {1}."
)


def _num_dendrite_out_of_range_repr(mode: Literal["ANN", "SNN"]) -> str:
    if mode == "ANN":
        max_limit = HwParams.N_DENDRITE_MAX_ANN
    else:
        max_limit = HwParams.N_DENDRITE_MAX_SNN

    return NUM_DENDRITE_OUT_OF_RANGE_TEXT.format(mode, max_limit)


class CoreParams(BaseModel):
    """Parameter model of register parameters listed in Section 2.4.1.

    NOTE: The parameters input in the model are declared in `docs/Table-of-Terms.md`.
    """

    model_config = ConfigDict(
        extra="ignore", validate_assignment=True, use_enum_values=True, strict=True
    )

    name: str = Field(
        frozen=True, description="Name of the physical core.", exclude=True
    )

    weight_precision: WeightPrecisionType = Field(
        frozen=True,
        serialization_alias="weight_width",
        description="Weight precision of crossbar.",
    )

    lcn_extension: LCNExtensionType = Field(
        frozen=True,
        serialization_alias="LCN",
        description="Scale of fan-in extension.",
    )

    input_width_format: InputWidthFormatType = Field(
        frozen=True,
        serialization_alias="input_width",
        description="Format of input spike.",
    )

    spike_width_format: SpikeWidthFormatType = Field(
        frozen=True,
        serialization_alias="spike_width",
        description="Format of output spike.",
    )

    num_dendrite: NonNegativeInt = Field(
        frozen=True,
        le=NUM_DENDRITE_MAX,
        serialization_alias="neuron_num",
        description="The number of used dendrites.",
    )

    max_pooling_en: MaxPoolingEnableType = Field(
        serialization_alias="pool_max",
        description="Enable max pooling or not in 8-bit input format.",
    )

    tick_wait_start: NonNegativeInt = Field(
        frozen=True,
        le=TICK_WAIT_START_MAX,
        description="The core begins to work at #N sync_all. 0 for not starting while 1 for staring forever.",
    )

    tick_wait_end: NonNegativeInt = Field(
        frozen=True,
        le=TICK_WAIT_END_MAX,
        description="The core keeps working within #N sync_all. 0 for not stopping.",
    )

    snn_mode_en: SNNModeEnableType = Field(
        frozen=True, serialization_alias="snn_en", description="Enable SNN mode or not."
    )

    target_lcn: LCNExtensionType = Field(
        frozen=True,
        serialization_alias="target_LCN",
        description="LCN extension of the core.",
    )

    test_chip_addr: Coord = Field(
        frozen=True, description="Destination address of output test frames."
    )

    @model_validator(mode="after")
    def _neuron_num_range_limit(self):
        _core_mode = get_core_mode(
            self.input_width_format, self.spike_width_format, self.snn_mode_en
        )
        if _core_mode.is_snn:
            if self.num_dendrite > HwParams.N_DENDRITE_MAX_SNN:
                raise ValueError(_num_dendrite_out_of_range_repr("SNN"))
        else:
            if self.num_dendrite > HwParams.N_DENDRITE_MAX_ANN:
                raise ValueError(_num_dendrite_out_of_range_repr("ANN"))

        return self

    @model_validator(mode="after")
    def _max_pooling_en_check(self):
        # XXX: If this parameter doesn't affect anything, this check can be removed &
        # set the entire model frozen=True.
        if (
            self.input_width_format is InputWidthFormatType.WIDTH_1BIT
            and self.max_pooling_en is MaxPoolingEnableType.ENABLE
        ):
            self.max_pooling_en = MaxPoolingEnableType.DISABLE

        return self

    @field_serializer("test_chip_addr")
    def _test_chip_addr(self, test_chip_addr: Coord) -> int:
        return test_chip_addr.address


ParamsReg = CoreParams


class _ParamsRegDict(TypedDict):
    """Typed dictionary of `ParamsReg` for typing check."""

    weight_width: int
    LCN: int
    input_width: L[0, 1]
    spike_width: L[0, 1]
    neuron_num: NonNegativeInt
    pool_max: L[0, 1]
    tick_wait_start: NonNegativeInt
    tick_wait_end: NonNegativeInt
    snn_en: L[0, 1]
    target_LCN: NonNegativeInt
    test_chip_addr: NonNegativeInt


ParamsRegChecker = TypeAdapter(_ParamsRegDict)
