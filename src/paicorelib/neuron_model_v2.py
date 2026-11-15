"""PAICORE 2.5 Pydantic neuron contracts for config-frame-3 inputs.

Use ``model_validate(mapping, strict=True)`` and ``model_dump()`` for toolchain
mappings; strict enum fields expect members, then serialize as integer codes.
Destination models validate assignments; attribute models do not. FP32/BF16
fields are numeric carriers, with BF16 reduction deferred to frame packing.
"""

from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    TypeAdapter,
    model_validator,
)

from .float_codec import BF16Param, FP32Param
from .neuron_defs import ResetMode
from .neuron_defs_v2 import (
    FoldType,
    LateralInhibitionMode,
    LeakAddMode,
    LeakMultiComparisonOrder,
    LeakMultiInputMode,
    LeakMultiMode,
    NeuronType,
    OfflineNeuRegLimV2,
    OnlineNeuRegLimV2,
    OnlineOutputType,
    OutputType,
    ThresholdNegMode,
    ThresholdPosMode,
    WeightCompressType,
)
from .neuron_model import NeuAttrs  # Reuse the base model

__all__ = [
    "NeuDestInfoV2",
    "OfflineNeuDestInfoV2",
    "OfflineNeuFullAttrsV2",
    "OfflineNeuFullAttrsV2Part1",
    "OfflineNeuFullAttrsV2Part2",
    "OfflineNeuHalfAttrsV2",
    "OnlineNeuDestInfoV2",
    "OnlineNeuHalfAttrsV2",
    "OnlineNeuFullAttrsV2",
    "OnlineNeuFullAttrsV2Part1",
    "OnlineNeuFullAttrsV2Part2",
    "OfflineNeuFoldedAttrsV2Part1",
    "OfflineNeuFoldedAttrsV2Part2",
    "OnlineNeuFoldedAttrsV2Part1",
    "OnlineNeuFoldedAttrsV2Part2",
    "OfflineNeuFullConfV2",
    "OfflineNeuHalfConfV2",
    "OnlineNeuFullConfV2",
    "OnlineNeuHalfConfV2",
]


class NeuDestInfoV2(BaseModel):
    """Relative output route, multicast region, timestamp and axon index.

    Core/copy coordinates are signed diagonal/X/Y offsets, not encoded bits.
    ``tick_relative`` is an eight-bit relative timestamp; the caller aligns it
    with target LCN. Axons are nine-bit offline and eight-bit online.
    """

    model_config = ConfigDict(
        extra="ignore", validate_assignment=True, use_enum_values=True
    )

    tick_relative: Annotated[
        NonNegativeInt,
        Field(
            le=OfflineNeuRegLimV2.TICK_RELATIVE_MAX,
            description="Relative time information.",
        ),
    ]
    addr_axon: Annotated[
        NonNegativeInt,
        Field(le=OfflineNeuRegLimV2.ADDR_AXON_MAX, description="Target axon address."),
    ]
    addr_core_xy: Annotated[
        int,
        Field(
            ge=OfflineNeuRegLimV2.ADDR_CORE_COORD_MIN,
            le=OfflineNeuRegLimV2.ADDR_CORE_COORD_MAX,
            description="Target core relative XY address.",
        ),
    ]
    addr_core_x: Annotated[
        int,
        Field(
            ge=OfflineNeuRegLimV2.ADDR_CORE_COORD_MIN,
            le=OfflineNeuRegLimV2.ADDR_CORE_COORD_MAX,
            description="Target core relative X address.",
        ),
    ]
    addr_core_y: Annotated[
        int,
        Field(
            ge=OfflineNeuRegLimV2.ADDR_CORE_COORD_MIN,
            le=OfflineNeuRegLimV2.ADDR_CORE_COORD_MAX,
            description="Target core relative Y address.",
        ),
    ]
    addr_copy_xy: Annotated[
        int,
        Field(
            ge=OfflineNeuRegLimV2.ADDR_CORE_COORD_MIN,
            le=OfflineNeuRegLimV2.ADDR_CORE_COORD_MAX,
            description="Number of copies in XY direction.",
        ),
    ]
    addr_copy_x: Annotated[
        int,
        Field(
            ge=OfflineNeuRegLimV2.ADDR_CORE_COORD_MIN,
            le=OfflineNeuRegLimV2.ADDR_CORE_COORD_MAX,
            description="Number of copies in X direction.",
        ),
    ]
    addr_copy_y: Annotated[
        int,
        Field(
            ge=OfflineNeuRegLimV2.ADDR_CORE_COORD_MIN,
            le=OfflineNeuRegLimV2.ADDR_CORE_COORD_MAX,
            description="Number of copies in Y direction.",
        ),
    ]


class _NeuCommonAttrsV2(NeuAttrs):
    weight_skew: Annotated[
        NonNegativeInt,
        Field(
            le=OfflineNeuRegLimV2.WEIGHT_SKEW_MAX,
            description="Vertical offset of neuron corresponding weight.",
        ),
    ]
    weight_address_start: Annotated[
        NonNegativeInt,
        Field(
            le=OfflineNeuRegLimV2.WEIGHT_ADDRESS_MAX,
            description="Start address of neuron corresponding weight SRAM.",
        ),
    ]
    weight_address_end: Annotated[
        NonNegativeInt,
        Field(
            le=OfflineNeuRegLimV2.WEIGHT_ADDRESS_MAX,
            description="End address of neuron corresponding weight SRAM.",
        ),
    ]

    fold_type: Annotated[FoldType, Field(description="Fold type selection.")]
    neuron_type: Annotated[NeuronType, Field(description="Neuron type selection.")]


class OfflineNeuDestInfoV2(NeuDestInfoV2):
    """Offline output destination with axon indices in [0, 511]."""

    pass


class OfflineNeuCommonAttrsV2(_NeuCommonAttrsV2):
    """Offline half-neuron fields, also exported as OfflineNeuFullAttrsV2Part1.

    ``vjt`` is an integer; the packer selects hardware bits without a model range
    check. Add OfflineNeuFullAttrsV2Part2 for the full-neuron layout.
    """

    output_type: Annotated[OutputType, Field(description="Output type selection.")]
    vjt: Annotated[
        int, Field(default=0, description="Current time step membrane potential.")
    ] = 0


OfflineNeuHalfAttrsV2 = OfflineNeuCommonAttrsV2
OfflineNeuFullAttrsV2Part1 = OfflineNeuHalfAttrsV2


class OnlineNeuDestInfoV2(NeuDestInfoV2):
    """Online output destination with axon indices in [0, 255]."""

    addr_axon: Annotated[
        NonNegativeInt,
        Field(le=OnlineNeuRegLimV2.ADDR_AXON_MAX, description="Target axon address."),
    ]


class OnlineNeuCommonAttrsV2(_NeuCommonAttrsV2):
    """Online half-neuron fields, also exported as OnlineNeuFullAttrsV2Part1.

    ``vjt`` is a numeric FP32 potential. Add OnlineNeuFullAttrsV2Part2 for a full
    neuron; do not reuse offline integer attributes for online numeric fields.
    """

    output_type: Annotated[
        OnlineOutputType, Field(description="Output type selection.")
    ]
    vjt: Annotated[
        FP32Param,
        Field(default=0.0, description="Current time step membrane potential."),
    ] = 0.0


OnlineNeuHalfAttrsV2 = OnlineNeuCommonAttrsV2
OnlineNeuFullAttrsV2Part1 = OnlineNeuHalfAttrsV2


class OfflineNeuFullAttrsV2Part2(NeuAttrs):
    """Offline full-neuron threshold/reset/leak fields, paired with part 1.

    Thresholds lack explicit model range checks; frame packing selects their
    low hardware bits. Other signed potential/shift bounds follow Field limits.
    """

    reset_mode: Annotated[ResetMode, Field(description="Reset mode selection.")]
    reset_v: Annotated[
        int,
        Field(
            ge=OfflineNeuRegLimV2.RESET_V_MIN,
            le=OfflineNeuRegLimV2.RESET_V_MAX,
            description="Membrane potential reset value.",
        ),
    ]

    threshold_neg_mode: Annotated[
        ThresholdNegMode, Field(description="Negative threshold mode selection.")
    ]
    threshold_pos_mode: Annotated[
        ThresholdPosMode, Field(description="Positive threshold mode selection.")
    ]

    threshold_neg: Annotated[int, Field(description="Negative threshold.")]
    threshold_pos: Annotated[int, Field(description="Positive threshold.")]

    lateral_inhibition: Annotated[
        LateralInhibitionMode, Field(description="Lateral inhibition mode selection.")
    ]
    leak_multi_sequence: Annotated[
        LeakMultiComparisonOrder, Field(description="Multiplicative leak sequence.")
    ]
    leak_multi_input: Annotated[
        LeakMultiInputMode,
        Field(description="Whether input participates in multiplicative leak."),
    ]
    leak_multi_mode: Annotated[
        LeakMultiMode, Field(description="Multiplicative leak mode selection.")
    ]
    leak_add_mode: Annotated[
        LeakAddMode, Field(description="Additive leak mode selection.")
    ]

    leak_tau: Annotated[
        int,
        Field(
            ge=OfflineNeuRegLimV2.LEAK_TAU_MIN,
            le=OfflineNeuRegLimV2.LEAK_TAU_MAX,
            description="Multiplicative leak shift amount.",
        ),
    ]
    leak_v: Annotated[
        int,
        Field(
            ge=OfflineNeuRegLimV2.LEAK_V_MIN,
            le=OfflineNeuRegLimV2.LEAK_V_MAX,
            description="Additive leak potential.",
        ),
    ]

    weight_compress: Annotated[
        WeightCompressType,
        Field(
            default=WeightCompressType.DENSE,
            description="Weight compression type (Dense/Sparse).",
        ),
    ]

    vjt_initial: Annotated[
        int,
        Field(
            default=0,
            ge=OfflineNeuRegLimV2.VJT_INITIAL_MIN,
            le=OfflineNeuRegLimV2.VJT_INITIAL_MAX,
            description="Initial membrane potential.",
        ),
    ] = 0


class OfflineNeuFullAttrsV2(OfflineNeuFullAttrsV2Part1, OfflineNeuFullAttrsV2Part2):
    """Combined offline full-neuron fields; usable for either part-specific input."""

    pass


class OnlineNeuFullAttrsV2Part2(NeuAttrs):
    """Online full-neuron reset/leak/initial potentials are numeric BF16 carriers.

    Thresholds are numeric FP32 carriers; supply numbers rather than payload bits.
    """

    reset_mode: Annotated[ResetMode, Field(description="Reset mode selection.")]
    reset_v: Annotated[BF16Param, Field(description="Membrane potential reset value.")]

    threshold_neg_mode: Annotated[
        ThresholdNegMode, Field(description="Negative threshold mode selection.")
    ]
    threshold_pos_mode: Annotated[
        ThresholdPosMode, Field(description="Positive threshold mode selection.")
    ]

    threshold_neg: Annotated[FP32Param, Field(description="Negative threshold.")]
    threshold_pos: Annotated[FP32Param, Field(description="Positive threshold.")]

    lateral_inhibition: Annotated[
        LateralInhibitionMode, Field(description="Lateral inhibition mode selection.")
    ]
    leak_multi_sequence: Annotated[
        LeakMultiComparisonOrder, Field(description="Multiplicative leak sequence.")
    ]
    leak_multi_input: Annotated[
        LeakMultiInputMode,
        Field(description="Whether input participates in multiplicative leak."),
    ]
    leak_multi_mode: Annotated[
        LeakMultiMode, Field(description="Multiplicative leak mode selection.")
    ]
    leak_add_mode: Annotated[
        LeakAddMode, Field(description="Additive leak mode selection.")
    ]

    leak_tau: Annotated[
        int,
        Field(
            ge=OnlineNeuRegLimV2.LEAK_TAU_MIN,
            le=OnlineNeuRegLimV2.LEAK_TAU_MAX,
            description="Multiplicative leak shift amount.",
        ),
    ]
    leak_v: Annotated[BF16Param, Field(description="Additive leak potential.")]

    weight_compress: Annotated[
        WeightCompressType,
        Field(
            default=WeightCompressType.DENSE,
            description="Weight compression type (Dense/Sparse).",
        ),
    ]
    vjt_initial: Annotated[
        BF16Param,
        Field(default=0.0, description="Initial membrane potential."),
    ] = 0.0


class OnlineNeuFullAttrsV2(OnlineNeuFullAttrsV2Part1, OnlineNeuFullAttrsV2Part2):
    """Combined online full-neuron fields; usable for either part-specific input."""

    pass


class _NeuFoldedAttrsV2Part1(NeuAttrs):
    """Fold geometry; fold_number must equal the three range fields multiplied."""

    fold_range_xy: Annotated[
        NonNegativeInt,
        Field(
            le=OfflineNeuRegLimV2.FOLD_RANGE_MAX, description="XY dimension fold width."
        ),
    ]
    fold_range_x: Annotated[
        NonNegativeInt,
        Field(
            le=OfflineNeuRegLimV2.FOLD_RANGE_MAX, description="X dimension fold width."
        ),
    ]
    fold_range_y: Annotated[
        NonNegativeInt,
        Field(
            le=OfflineNeuRegLimV2.FOLD_RANGE_MAX, description="Y dimension fold width."
        ),
    ]

    fold_skew_xy: Annotated[
        NonNegativeInt,
        Field(
            le=OfflineNeuRegLimV2.FOLD_SKEW_MAX,
            description="XY dimension weight offset step.",
        ),
    ]
    fold_skew_x: Annotated[
        NonNegativeInt,
        Field(
            le=OfflineNeuRegLimV2.FOLD_SKEW_MAX,
            description="X dimension weight offset step.",
        ),
    ]
    fold_skew_y: Annotated[
        NonNegativeInt,
        Field(
            le=OfflineNeuRegLimV2.FOLD_SKEW_MAX,
            description="Y dimension weight offset step.",
        ),
    ]

    fold_axon_xy: Annotated[
        NonNegativeInt,
        Field(
            le=OfflineNeuRegLimV2.FOLD_AXON_MAX,
            description="XY dimension Axon address offset step.",
        ),
    ]
    fold_axon_x: Annotated[
        NonNegativeInt,
        Field(
            le=OfflineNeuRegLimV2.FOLD_AXON_MAX,
            description="X dimension Axon address offset step.",
        ),
    ]
    fold_axon_y: Annotated[
        NonNegativeInt,
        Field(
            le=OfflineNeuRegLimV2.FOLD_AXON_MAX,
            description="Y dimension Axon address offset step.",
        ),
    ]

    fold_number: Annotated[
        NonNegativeInt,
        Field(
            le=OfflineNeuRegLimV2.FOLD_NUMBER_MAX,
            description="Total number of folded neurons.",
        ),
    ]

    @model_validator(mode="after")
    def check_fold_number(self):
        if (
            self.fold_range_xy * self.fold_range_x * self.fold_range_y
            != self.fold_number
        ):
            raise ValueError(
                "'fold_number' must be equal to 'fold_range_xy' * 'fold_range_x' * 'fold_range_y', "
                f"but {self.fold_number} != {self.fold_range_xy} * {self.fold_range_x} * {self.fold_range_y}"
            )

        return self


class OfflineNeuFoldedAttrsV2Part1(_NeuFoldedAttrsV2Part1):
    """Offline fold geometry; caller separately supplies the folded-state sequence."""

    pass


class OnlineNeuFoldedAttrsV2Part1(_NeuFoldedAttrsV2Part1):
    """Online fold geometry; pair with online FP32 folded-state records."""

    pass


class OfflineNeuFoldedAttrsV2Part2(NeuAttrs):
    """Four integer folded potentials, packed in index order 0..3 as low 32-bit values."""

    fold_vjt_3: Annotated[
        int, Field(default=0, description="Folded neuron 3 membrane potential.")
    ] = 0
    fold_vjt_2: Annotated[
        int, Field(default=0, description="Folded neuron 2 membrane potential.")
    ] = 0
    fold_vjt_1: Annotated[
        int, Field(default=0, description="Folded neuron 1 membrane potential.")
    ] = 0
    fold_vjt_0: Annotated[
        int, Field(default=0, description="Folded neuron 0 membrane potential.")
    ] = 0


class OnlineNeuFoldedAttrsV2Part2(NeuAttrs):
    """Four numeric FP32 folded potentials, packed in index order 0..3."""

    fold_vjt_3: Annotated[
        FP32Param,
        Field(default=0.0, description="Folded neuron 3 membrane potential."),
    ] = 0.0
    fold_vjt_2: Annotated[
        FP32Param,
        Field(default=0.0, description="Folded neuron 2 membrane potential."),
    ] = 0.0
    fold_vjt_1: Annotated[
        FP32Param,
        Field(default=0.0, description="Folded neuron 1 membrane potential."),
    ] = 0.0
    fold_vjt_0: Annotated[
        FP32Param,
        Field(default=0.0, description="Folded neuron 0 membrane potential."),
    ] = 0.0


# Neuron configuration
class OfflineNeuFullConfV2(BaseModel):
    attrs: OfflineNeuFullAttrsV2
    dest_info: OfflineNeuDestInfoV2


class OfflineNeuHalfConfV2(BaseModel):
    attrs: OfflineNeuHalfAttrsV2
    dest_info: OfflineNeuDestInfoV2


class OnlineNeuFullConfV2(BaseModel):
    attrs: OnlineNeuFullAttrsV2
    dest_info: OnlineNeuDestInfoV2


class OnlineNeuHalfConfV2(BaseModel):
    attrs: OnlineNeuHalfAttrsV2
    dest_info: OnlineNeuDestInfoV2


OfflineNeuDestInfoV2Checker = TypeAdapter(OfflineNeuDestInfoV2)
OfflineNeuHalfAttrsV2Checker = TypeAdapter(OfflineNeuHalfAttrsV2)
OfflineNeuFullAttrsV2Part2Checker = TypeAdapter(OfflineNeuFullAttrsV2Part2)
OnlineNeuDestInfoV2Checker = TypeAdapter(OnlineNeuDestInfoV2)
OnlineNeuHalfAttrsV2Checker = TypeAdapter(OnlineNeuHalfAttrsV2)
OnlineNeuFullAttrsV2Checker = TypeAdapter(OnlineNeuFullAttrsV2)
OnlineNeuFullAttrsV2Part2Checker = TypeAdapter(OnlineNeuFullAttrsV2Part2)
OfflineNeuFoldedAttrsV2Part1Checker = TypeAdapter(OfflineNeuFoldedAttrsV2Part1)
OfflineNeuFoldedAttrsV2Part2Checker = TypeAdapter(OfflineNeuFoldedAttrsV2Part2)
OnlineNeuFoldedAttrsV2Part1Checker = TypeAdapter(OnlineNeuFoldedAttrsV2Part1)
OnlineNeuFoldedAttrsV2Part2Checker = TypeAdapter(OnlineNeuFoldedAttrsV2Part2)
