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

__all__ = ["OfflineCoreReg", "ParamsReg"]

NUM_DENDRITE_BIT_MAX = 13  # Unsigned
TICK_WAIT_START_BIT_MAX = 15  # Unsigned
TICK_WAIT_END_BIT_MAX = 15  # Unsigned

TICK_WAIT_START_MAX = _mask(TICK_WAIT_START_BIT_MAX)
TICK_WAIT_END_MAX = _mask(TICK_WAIT_END_BIT_MAX)

NUM_DENDRITE_OUT_OF_RANGE_TEXT = (
    "parameter 'num_dendrite' out of range. When input width is {0}-bit, "
    + "the number of valid dendrites should be <= {1}."
)

_N_REPEAT_NRAM_UNSET = 0


class _CoreReg(BaseModel):
    """Parameter model of registers of cores, listed in Section 2.4.1.

    NOTE: The parameters in the model are declared in `docs/Table-of-Terms.md`.
    """

    model_config = ConfigDict(
        extra="ignore", validate_assignment=True, use_enum_values=True, strict=True
    )

    name: str = Field(
        frozen=True, description="Name of the physical core.", exclude=True
    )

    test_chip_addr: Coord = Field(
        frozen=True, description="Destination address of output test frames."
    )

    @field_serializer("test_chip_addr")
    def _test_chip_addr(self, test_chip_addr: Coord) -> int:
        return test_chip_addr.address


class OfflineCoreReg(_CoreReg):
    """Parameter model of registers of cores, listed in Section 2.4.1.

    NOTE: The parameters in the model are declared in `docs/Table-of-Terms.md`.
    """

    weight_width: WeightWidth = Field(
        frozen=True, description="Weight bit width of the crossbar."
    )

    lcn_extension: LCN_EX = Field(
        frozen=True,
        serialization_alias="LCN",
        description="Scale of fan-in extension of the core.",
    )

    input_width_format: InputWidthFormat = Field(
        frozen=True,
        serialization_alias="input_width",
        description="Width of input data.",
    )

    spike_width_format: SpikeWidthFormat = Field(
        frozen=True,
        serialization_alias="spike_width",
        description="Width of output data.",
    )

    # will be checked in its model_validator.
    num_dendrite: NonNegativeInt = Field(
        frozen=True,
        description="The number of valid dendrites in the core.",
    )

    max_pooling_en: MaxPoolingEnable = Field(
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

    snn_mode_en: SNNModeEnable = Field(
        frozen=True, serialization_alias="snn_en", description="Enable SNN mode or not."
    )

    target_lcn: LCN_EX = Field(
        frozen=True,
        serialization_alias="target_LCN",
        description="The rate of the fan-in extension of the target cores.",
    )

    n_repeat_nram: NonNegativeInt = Field(
        default=_N_REPEAT_NRAM_UNSET,
        description="The number of repetitions that need to be repeated for neurons to be placed within the NRAM.",
    )

    @model_validator(mode="after")
    def _dendrite_num_range_limit(self):
        _core_mode = get_core_mode(
            self.input_width_format, self.spike_width_format, self.snn_mode_en
        )
        if _core_mode.is_iw8:
            if self.num_dendrite > HwParams.N_DENDRITE_MAX_ANN:
                raise ValueError(
                    NUM_DENDRITE_OUT_OF_RANGE_TEXT.format(
                        8, HwParams.N_DENDRITE_MAX_ANN
                    )
                )
        else:
            if self.num_dendrite > HwParams.N_DENDRITE_MAX_SNN:
                raise ValueError(
                    NUM_DENDRITE_OUT_OF_RANGE_TEXT.format(
                        1, HwParams.N_DENDRITE_MAX_SNN
                    )
                )

        return self

    @model_validator(mode="after")
    def _max_pooling_disable_iw1(self):
        if self.input_width_format is InputWidthFormat.WIDTH_1BIT:
            self.pool_max = MaxPoolingEnable.DISABLE

        return self

    @model_validator(mode="after")
    def _n_repeat_nram_unset(self):
        """In case 'n_repeat_nram' is unset, calculate it."""
        if self.n_repeat_nram == _N_REPEAT_NRAM_UNSET:
            # Since config 'use_enum_values' is enabled, 'input_width_format' is now an integer
            # after validation.
            if self.input_width_format == 0:  # 1-bit
                # dendrite_comb_rate = lcn + ww
                self.n_repeat_nram = 1 << (self.lcn_extension + self.weight_width)
            else:
                self.n_repeat_nram = 1

        return self


ParamsReg = OfflineCoreReg


class _OfflineCoreRegDict(TypedDict):
    """Typed dictionary of `OfflineCoreReg` for typing check. Use the following keys as the  \
        serialization name of the parametric model above.
    """

    weight_width: int
    LCN: int
    input_width: Literal[0, 1]
    spike_width: Literal[0, 1]
    num_dendrite: NonNegativeInt
    pool_max: Literal[0, 1]
    tick_wait_start: NonNegativeInt
    tick_wait_end: NonNegativeInt
    snn_en: Literal[0, 1]
    target_LCN: NonNegativeInt
    test_chip_addr: NonNegativeInt


CoreRegChecker = TypeAdapter(_OfflineCoreRegDict)
