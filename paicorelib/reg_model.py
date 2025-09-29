import random
import string
from typing import Annotated, Any

import numpy as np
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PositiveInt,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic.types import NonNegativeInt

from .coordinate import ChipCoord, Coord, CoordAddr, to_coord
from .hw_defs import HwOfflineCoreParams as OffCoreParams
from .hw_defs import HwOnlineCoreParams as OnCoreParams
from .ram_defs import RAMDefs
from .reg_defs import OnlineRegDefs as OnRegDefs
from .reg_defs import *
from .reg_defs import RegDefs

__all__ = ["CoreReg", "OfflineCoreReg", "OnlineCoreReg"]

NUM_DENDRITE_OUT_OF_RANGE_TEXT = (
    "parameter 'num_dendrite' out of range. When input width is {0}-bit, "
    + "the number of valid dendrites should be <= {1}."
)

N_REPEAT_NRAM_UNSET = 0


def _gen_random_str(length: int = 8) -> str:
    chars = string.ascii_letters + string.digits
    return "CoreReg_" + "".join(random.choice(chars) for _ in range(length))


class CoreReg(BaseModel):
    """Parameter model of registers of cores, listed in Section 2.4.1.

    NOTE: The parameters in the model are declared in `docs/Table-of-Terms.md`.
    """

    model_config = ConfigDict(
        extra="ignore", validate_assignment=True, use_enum_values=True, strict=True
    )

    name: Annotated[
        str,
        Field(
            default_factory=lambda: _gen_random_str(),
            frozen=True,
            description="Name of the physical core.",
            exclude=True,
        ),
    ]


def _to_coord_check(v: Any) -> ChipCoord:
    if not isinstance(v, (Coord, CoordAddr, tuple)):
        raise TypeError(
            f"parameter 'test_chip_addr' should be a CoordLike, but got {type(v)}"
        )

    return to_coord(v)


class OfflineCoreReg(CoreReg):
    """Parameter model of registers of cores, listed in Section 2.4.1.

    NOTE: The parameters in the model are declared in `docs/Table-of-Terms.md`.
    """

    weight_width: Annotated[
        WeightWidth, Field(frozen=True, description="Weight bit width of the crossbar.")
    ]

    lcn: Annotated[
        LCN_EX,
        Field(frozen=True, description="Scale of fan-in extension of the core."),
    ]

    input_width: Annotated[
        InputWidthFormat,
        Field(frozen=True, description="Width of input data."),
    ]

    spike_width: SpikeWidthFormat = Field(
        frozen=True, description="Width of output data."
    )

    num_dendrite: NonNegativeInt = Field(
        frozen=True,
        serialization_alias="neuron_num",
        description="The number of valid dendrites in the core.",
    )

    max_pooling_en: MaxPoolingEnable = Field(
        serialization_alias="pool_max",
        description="Enable max pooling or not in 8-bit input format.",
    )

    tick_wait_start: Annotated[
        NonNegativeInt,
        Field(
            frozen=True,
            le=RegDefs.TICK_WAIT_START_MAX,
            description="The core begins to work at #N synchronization signal. 0 for not starting   \
                while 1 for staring forever.",
        ),
    ]

    tick_wait_end: Annotated[
        NonNegativeInt,
        Field(
            frozen=True,
            le=RegDefs.TICK_WAIT_END_MAX,
            description="The core keeps working within #N synchronization signal. 0 for not stopping.",
        ),
    ]

    snn_en: SNNModeEnable = Field(frozen=True, description="Enable SNN mode or not.")

    target_lcn: LCN_EX = Field(
        frozen=True,
        description="The rate of the fan-in extension of the target cores.",
    )

    test_chip_addr: Annotated[
        ChipCoord,
        Field(frozen=True, description="Destination address of output test frames."),
        BeforeValidator(_to_coord_check),
    ]

    n_repeat_nram: NonNegativeInt = Field(
        default=N_REPEAT_NRAM_UNSET,
        description="The number of repetitions that need to be repeated for neurons to be placed within the NRAM.",
    )

    @model_validator(mode="after")
    def dendrite_num_range_limit(self):
        _core_mode = get_core_mode(self.input_width, self.spike_width, self.snn_en)
        if _core_mode.is_iw8:
            if self.num_dendrite > OffCoreParams.N_DENDRITE_MAX_ANN:
                raise ValueError(
                    NUM_DENDRITE_OUT_OF_RANGE_TEXT.format(
                        8, OffCoreParams.N_DENDRITE_MAX_ANN
                    )
                )
        else:
            if self.num_dendrite > OffCoreParams.N_DENDRITE_MAX_SNN:
                raise ValueError(
                    NUM_DENDRITE_OUT_OF_RANGE_TEXT.format(
                        1, OffCoreParams.N_DENDRITE_MAX_SNN
                    )
                )

        return self

    @model_validator(mode="after")
    def max_pooling_disable_iw1(self):
        if (
            self.input_width == InputWidthFormat.WIDTH_1BIT
            and self.max_pooling_en == MaxPoolingEnable.ENABLE
        ):
            self.max_pooling_en = MaxPoolingEnable.DISABLE

        return self

    @field_serializer("test_chip_addr")
    def _test_chip_addr(self, test_chip_addr: Coord) -> int:
        return test_chip_addr.address

    @model_validator(mode="after")
    def n_repeat_nram_unset(self):
        """In case 'n_repeat_nram' is unset, calculate it."""
        if self.n_repeat_nram == N_REPEAT_NRAM_UNSET:
            # Since config 'use_enum_values' is enabled, 'input_width_format' is now an integer
            # after validation.
            if self.input_width == 0:  # 1-bit
                # dendrite_comb_rate = lcn + ww
                self.n_repeat_nram = 1 << (self.lcn + self.weight_width)
            else:
                self.n_repeat_nram = 1

        return self


LUT_RANDOM_EN_LEN = OnCoreParams.LUT_LEN
COORD_MAX = RAMDefs.COORD_MAX


def inhi_rid_range_check(rid: int) -> int:
    # 11100~11111 means the rid <= 0b00011
    if rid >= 0b100:
        raise ValueError(
            f"parameter 'inhi_core_x/y_ex' should be less than 4, but got {rid}"
        )

    return rid


def lut_random_en_check(v: Any) -> int:
    if not isinstance(v, (list, np.ndarray)):
        raise TypeError(
            f"parameter 'lut_random_en' should be a list or numpy array, but got {type(v)}"
        )

    v_arr = np.asarray(v, dtype=np.bool)
    if (l := v_arr.size) != LUT_RANDOM_EN_LEN:
        raise ValueError(
            f"the length of 'lut_random_en' should be {LUT_RANDOM_EN_LEN}, but got {l}"
        )

    bitmap = np.sum(v_arr * (1 << np.arange(LUT_RANDOM_EN_LEN)))
    return int(bitmap)


class OnlineCoreReg(CoreReg):
    """Parameter model of registers of online cores, listed in Section 2.5.1.

    NOTE: The parameters in the model are declared in `docs/Table-of-Terms.md`.
    """

    model_config = CoreReg.model_config | {"frozen": True}

    weight_width: Annotated[
        WeightWidth,
        Field(
            serialization_alias="bit_select",
            description="Weight bit width of the crossbar.",
        ),
    ]

    lcn: Annotated[
        LCN_EX,
        Field(
            serialization_alias="group_select",
            description="Scale of fan-in extension of the core.",
        ),
    ]

    lateral_inhi_value: Annotated[
        int,
        Field(
            ge=OnRegDefs.LATERAL_INHI_VALUE_MIN,
            le=OnRegDefs.LATERAL_INHI_VALUE_MAX,
            description="The value of the lateral inhibition.",
        ),
    ]

    weight_decay_value: Annotated[
        int,
        Field(
            ge=OnRegDefs.WEIGHT_DECAY_VALUE_MIN,
            le=OnRegDefs.WEIGHT_DECAY_VALUE_MAX,
            description="The value of the weight decay.",
        ),
    ]

    upper_weight: Annotated[
        int,
        Field(
            ge=OnRegDefs.UPPER_WEIGHT_MIN,
            le=OnRegDefs.UPPER_WEIGHT_MAX,
            description="The upper limit of the weight update.",
        ),
    ]

    lower_weight: Annotated[
        int,
        Field(
            ge=OnRegDefs.LOWER_WEIGHT_MIN,
            le=OnRegDefs.LOWER_WEIGHT_MAX,
            description="The lower limit of the weight update.",
        ),
    ]

    neuron_start: Annotated[
        NonNegativeInt,
        Field(
            le=OnRegDefs.NEU_START_MAX,
            description="The start address of the valid neuron in the NRAM.",
        ),
    ]

    neuron_end: Annotated[
        NonNegativeInt,
        Field(
            le=OnRegDefs.NEU_END_MAX,
            description="The end address of the valid neuron in the NRAM.",
        ),
    ]

    inhi_core_x_ex: Annotated[
        NonNegativeInt,
        Field(
            serialization_alias="inhi_core_x_star",
            description="X replication identifier of the inhibitory core.",
        ),
        AfterValidator(inhi_rid_range_check),
    ]

    inhi_core_y_ex: Annotated[
        NonNegativeInt,
        Field(
            serialization_alias="inhi_core_y_star",
            description="Y replication identifier of the inhibitory core.",
        ),
        AfterValidator(inhi_rid_range_check),
    ]

    tick_wait_start: Annotated[
        NonNegativeInt,
        Field(
            le=RegDefs.TICK_WAIT_START_MAX,
            serialization_alias="core_start_time",
            description="The core begins to work at #N synchronization signal. 0 for not starting   \
                while 1 for staring forever.",
        ),
    ]

    tick_wait_end: Annotated[
        NonNegativeInt,
        Field(
            le=RegDefs.TICK_WAIT_END_MAX,
            serialization_alias="core_hold_time",
            description="The core keeps working within #N synchronization signal. 0 for not stopping.",
        ),
    ]

    lut_random_en: Annotated[
        int,  # bitmap of `LUT_LEN` bits
        Field(description="Enable random update for LUT or not."),
        BeforeValidator(lut_random_en_check),
    ]

    decay_random_en: Annotated[
        DecayRandomEnable,
        Field(description="Enable random update for weight decay or not."),
    ]

    leak_order: Annotated[
        LeakOrder,
        Field(
            serialization_alias="leakage_order",
            description="Leak after comparison or before.",
        ),
    ]

    online_mode_en: Annotated[
        OnlineModeEnable,
        Field(description="Enable online mode or not (offline inference mode)."),
    ]

    test_chip_addr: Annotated[
        ChipCoord,
        Field(
            serialization_alias="test_address",
            description="Destination address of output test frames.",
        ),
        BeforeValidator(to_coord),
    ]

    random_seed: Annotated[
        PositiveInt,
        Field(le=OnRegDefs.RANDOM_SEED_MAX, description="The non-zero random seed."),
    ]

    @field_validator("lcn")
    @classmethod
    def lcn_check(cls, v: int) -> int:
        if v > LCN_EX.LCN_8X:
            raise ValueError(
                f"parameter 'lcn_ex' out of range {LCN_EX.LCN_8X}, but got {v}"
            )

        return v

    @model_validator(mode="after")
    def weight_update_range_check(self):
        if self.upper_weight < self.lower_weight:
            raise ValueError(
                f"parameter 'upper_weight' should be greater than or equal to 'lower_weight', but got "
                f"{self.upper_weight} < {self.lower_weight}"
            )

        return self

    @model_validator(mode="after")
    def valid_neuron_range_check(self):
        if self.neuron_start > self.neuron_end:
            raise ValueError(
                f"parameter 'neuron_start' should be less than or equal to 'neuron_end', but got "
                f"{self.neuron_start} > {self.neuron_end}"
            )

        return self

    @field_serializer("test_chip_addr")
    def _test_chip_addr(self, test_chip_addr: Coord) -> int:
        return test_chip_addr.address
