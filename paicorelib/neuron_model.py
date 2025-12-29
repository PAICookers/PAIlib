from typing import Annotated, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, model_validator

from .neuron_defs import (
    FoldType,
    LateralInhibitionMode,
    LeakAddMode,
    LeakMultiInputMode,
    LeakMultiMode,
    LeakMultiSequence,
    NeuronLim,
    NeuronType,
    OutputType,
    ResetMode,
    ThresholdNegMode,
    ThresholdPosMode,
    WeightCompressMode,
)

__all__ = [
    "NeuronPart1",
    "NeuronPart2",
    "FoldedNeuronPart1",
    "FoldedNeuronPart2",
]


class NeuronPart1(BaseModel):
    """Full Neuron Parameter Part 1 / Half Neuron Parameter (2.3.3)."""

    model_config = ConfigDict(
        extra="ignore", validate_assignment=True, use_enum_values=True, strict=True
    )

    tick_relative: Annotated[
        NonNegativeInt,
        Field(le=NeuronLim.TICK_RELATIVE_MAX, description="Relative time information."),
    ]
    addr_axon: Annotated[
        NonNegativeInt,
        Field(le=NeuronLim.ADDR_AXON_MAX, description="Target axon address."),
    ]

    # Target Core Addresses (Sign-Magnitude 6-bit: -31 to 31)
    addr_core_xy: Annotated[
        int,
        Field(
            ge=NeuronLim.ADDR_CORE_OFFSET_MIN,
            le=NeuronLim.ADDR_CORE_OFFSET_MAX,
            description="Target core relative XY address.",
        ),
    ]
    addr_core_x: Annotated[
        int,
        Field(
            ge=NeuronLim.ADDR_CORE_OFFSET_MIN,
            le=NeuronLim.ADDR_CORE_OFFSET_MAX,
            description="Target core relative X address.",
        ),
    ]
    addr_core_y: Annotated[
        int,
        Field(
            ge=NeuronLim.ADDR_CORE_OFFSET_MIN,
            le=NeuronLim.ADDR_CORE_OFFSET_MAX,
            description="Target core relative Y address.",
        ),
    ]

    # Broadcast Addresses (Sign-Magnitude 6-bit: -31 to 31)
    addr_copy_xy: Annotated[
        int,
        Field(
            ge=NeuronLim.ADDR_CORE_OFFSET_MIN,
            le=NeuronLim.ADDR_CORE_OFFSET_MAX,
            description="Target core XY broadcast address.",
        ),
    ]
    addr_copy_x: Annotated[
        int,
        Field(
            ge=NeuronLim.ADDR_CORE_OFFSET_MIN,
            le=NeuronLim.ADDR_CORE_OFFSET_MAX,
            description="Target core X broadcast address.",
        ),
    ]
    addr_copy_y: Annotated[
        int,
        Field(
            ge=NeuronLim.ADDR_CORE_OFFSET_MIN,
            le=NeuronLim.ADDR_CORE_OFFSET_MAX,
            description="Target core Y broadcast address.",
        ),
    ]

    weight_skew: Annotated[
        NonNegativeInt,
        Field(
            le=NeuronLim.WEIGHT_SKEW_MAX,
            description="Vertical offset of neuron corresponding weight.",
        ),
    ]
    weight_address_start: Annotated[
        NonNegativeInt,
        Field(
            le=NeuronLim.WEIGHT_ADDRESS_MAX,
            description="Start address of neuron corresponding weight SRAM.",
        ),
    ]
    weight_address_end: Annotated[
        NonNegativeInt,
        Field(
            le=NeuronLim.WEIGHT_ADDRESS_MAX,
            description="End address of neuron corresponding weight SRAM.",
        ),
    ]

    output_type: Annotated[OutputType, Field(description="Output type selection.")]
    fold_type: Annotated[FoldType, Field(description="Fold type selection.")]
    neuron_type: Annotated[NeuronType, Field(description="Neuron type selection.")]

    vjt: Annotated[int, Field(description="Current time step membrane potential.")]


class NeuronPart2(BaseModel):
    """Full Neuron Parameter Part 2 (2.3.3)."""

    model_config = ConfigDict(
        extra="ignore", validate_assignment=True, use_enum_values=True, strict=True
    )

    reset_mode: Annotated[ResetMode, Field(description="Reset mode selection.")]
    reset_v: Annotated[
        int,
        Field(
            ge=NeuronLim.RESET_V_MIN,
            le=NeuronLim.RESET_V_MAX,
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
    leakmulti_sequence: Annotated[
        LeakMultiSequence, Field(description="Multiplicative leak sequence.")
    ]
    leakmulti_input: Annotated[
        LeakMultiInputMode,
        Field(description="Whether input participates in multiplicative leak."),
    ]
    leakmulti_mode: Annotated[
        LeakMultiMode, Field(description="Multiplicative leak mode selection.")
    ]
    leak_add_mode: Annotated[
        LeakAddMode, Field(description="Additive leak mode selection.")
    ]

    leak_tau: Annotated[
        int,
        Field(
            ge=NeuronLim.LEAK_TAU_MIN,
            le=NeuronLim.LEAK_TAU_MAX,
            description="Multiplicative leak shift amount.",
        ),
    ]
    leak_v: Annotated[
        int,
        Field(
            ge=NeuronLim.LEAK_V_MIN,
            le=NeuronLim.LEAK_V_MAX,
            description="Additive leak potential.",
        ),
    ]

    weight_compress: Annotated[
        WeightCompressMode, Field(description="Weight type (Dense/Sparse).")
    ]
    vjt_initial: Annotated[
        int,
        Field(
            ge=NeuronLim.VJT_INITIAL_MIN,
            le=NeuronLim.VJT_INITIAL_MAX,
            description="Initial membrane potential.",
        ),
    ]


class FoldedNeuronPart1(BaseModel):
    """Folded Neuron Parameter Part 1 (2.3.3)."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True, strict=True)

    fold_range_xy: Annotated[
        NonNegativeInt,
        Field(le=NeuronLim.FOLD_RANGE_MAX, description="XY dimension fold width."),
    ]
    fold_range_x: Annotated[
        NonNegativeInt,
        Field(le=NeuronLim.FOLD_RANGE_MAX, description="X dimension fold width."),
    ]
    fold_range_y: Annotated[
        NonNegativeInt,
        Field(le=NeuronLim.FOLD_RANGE_MAX, description="Y dimension fold width."),
    ]

    fold_skew_xy: Annotated[
        NonNegativeInt,
        Field(
            le=NeuronLim.FOLD_SKEW_MAX, description="XY dimension weight offset step."
        ),
    ]
    fold_skew_x: Annotated[
        NonNegativeInt,
        Field(
            le=NeuronLim.FOLD_SKEW_MAX, description="X dimension weight offset step."
        ),
    ]
    fold_skew_y: Annotated[
        NonNegativeInt,
        Field(
            le=NeuronLim.FOLD_SKEW_MAX, description="Y dimension weight offset step."
        ),
    ]

    fold_axon_xy: Annotated[
        NonNegativeInt,
        Field(
            le=NeuronLim.FOLD_AXON_MAX,
            description="XY dimension Axon address offset step.",
        ),
    ]
    fold_axon_x: Annotated[
        NonNegativeInt,
        Field(
            le=NeuronLim.FOLD_AXON_MAX,
            description="X dimension Axon address offset step.",
        ),
    ]
    fold_axon_y: Annotated[
        NonNegativeInt,
        Field(
            le=NeuronLim.FOLD_AXON_MAX,
            description="Y dimension Axon address offset step.",
        ),
    ]

    fold_number: Annotated[
        NonNegativeInt,
        Field(
            le=NeuronLim.FOLD_NUMBER_MAX, description="Total number of folded neurons."
        ),
    ]

    @model_validator(mode="after")
    def check_fold_number(self) -> "FoldedNeuronPart1":
        if (
            self.fold_range_xy * self.fold_range_x * self.fold_range_y
            != self.fold_number
        ):
            raise ValueError(
                "fold_number must be equal to fold_range_xy * fold_range_x * fold_range_y"
            )
        return self


class FoldedNeuronPart2(BaseModel):
    """Folded Neuron Parameter Part 2 (2.3.3)."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True, strict=True)

    fold_vjt_3: Annotated[int, Field(description="Folded neuron 3 membrane potential.")]
    fold_vjt_2: Annotated[int, Field(description="Folded neuron 2 membrane potential.")]
    fold_vjt_1: Annotated[int, Field(description="Folded neuron 1 membrane potential.")]
    fold_vjt_0: Annotated[int, Field(description="Folded neuron 0 membrane potential.")]
