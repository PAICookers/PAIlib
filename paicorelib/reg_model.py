from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator
from pydantic.type_adapter import TypeAdapter
from pydantic.types import NonNegativeInt
from typing_extensions import TypedDict  # Use `typing_extensions.TypedDict`.

from .coordinate import Coord
from .framelib.frame_defs import _mask
from .hw_defs import HwParams
from .reg_types import *

__all__ = ["CoreParams", "ParamsReg"]

L = Literal

NUM_DENDRITE_BIT_MAX = 13
TICK_WAIT_START_BIT_MAX = 15
TICK_WAIT_END_BIT_MAX = 15
TARGET_LCN_BIT_MAX = 4
TEST_CHIP_ADDR_BIT_MAX = 10

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

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    name: str = Field(description="Name of the physical core.", exclude=True)

    weight_precision: WeightPrecisionType = Field(
        serialization_alias="weight_width",
        description="Weight precision of crossbar.",
    )

    lcn_extension: LCNExtensionType = Field(
        serialization_alias="LCN",
        description="Scale of fan-in extension.",
    )

    input_width_format: InputWidthFormatType = Field(
        serialization_alias="input_width", description="Format of input spike."
    )

    spike_width_format: SpikeWidthFormatType = Field(
        serialization_alias="spike_width", description="Format of output spike."
    )

    num_dendrite: NonNegativeInt = Field(
        le=_mask(NUM_DENDRITE_BIT_MAX),
        serialization_alias="neuron_num",
        description="The number of used dendrites.",
    )

    max_pooling_en: MaxPoolingEnableType = Field(
        serialization_alias="pool_max",
        description="Enable max pooling or not in 8-bit input format.",
    )

    tick_wait_start: NonNegativeInt = Field(
        le=_mask(TICK_WAIT_START_BIT_MAX),
        description="The core begins to work at #N sync_all. 0 for not starting while 1 for staring forever.",
    )

    tick_wait_end: NonNegativeInt = Field(
        le=_mask(TICK_WAIT_END_BIT_MAX),
        description="The core keeps working within #N sync_all. 0 for not stopping.",
    )

    snn_mode_en: SNNModeEnableType = Field(
        serialization_alias="snn_en", description="Enable SNN mode or not."
    )

    target_lcn: LCNExtensionType = Field(
        serialization_alias="target_LCN", description="LCN extension of the core."
    )

    test_chip_addr: Coord = Field(
        description="Destination address of output test frames."
    )

    """Parameter checks"""

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
        if (
            self.input_width_format is InputWidthFormatType.WIDTH_1BIT
            and self.max_pooling_en is MaxPoolingEnableType.ENABLE
        ):
            self.max_pooling_en = MaxPoolingEnableType.DISABLE

        return self

    """Parameter serializers"""

    @field_serializer("weight_precision")
    def _weight_precision(self, weight_precision: WeightPrecisionType) -> int:
        return weight_precision.value

    @field_serializer("lcn_extension")
    def _lcn_extension(self, lcn_extension: LCNExtensionType) -> int:
        return lcn_extension.value

    @field_serializer("input_width_format")
    def _input_width_format(self, input_width_format: InputWidthFormatType) -> int:
        return input_width_format.value

    @field_serializer("spike_width_format")
    def _spike_width_format(self, spike_width_format: SpikeWidthFormatType) -> int:
        return spike_width_format.value

    @field_serializer("max_pooling_en")
    def _max_pooling_en(self, max_pooling_en: MaxPoolingEnableType) -> int:
        return max_pooling_en.value

    @field_serializer("snn_mode_en")
    def _snn_mode_en(self, snn_mode_en: SNNModeEnableType) -> int:
        return snn_mode_en.value

    @field_serializer("target_lcn")
    def _target_lcn(self, target_lcn: LCNExtensionType) -> int:
        return target_lcn.value

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
