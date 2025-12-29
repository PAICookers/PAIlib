import json

import pytest
from pydantic import ValidationError

from paicorelib.neuron_defs import (
    FoldType,
    LateralInhibitionMode,
    LeakAddMode,
    LeakMultiInputMode,
    LeakMultiMode,
    LeakMultiComparisonOrder,
    NeuronLim,
    NeuronType,
    OutputType,
    ThresholdNegMode,
    ThresholdPosMode,
    WeightCompressMode,
)
from paicorelib.neuron_model import (
    FoldedNeuronParameter,
    FoldedNeuronPotential,
    NeuronDifferent,
    NeuronCommon,
    NeuronDestination2_5,
)

from paicorelib.ram_defs import ResetMode


class TestNeuronDestination2_5Model:
    @pytest.fixture
    def default_params(self):
        return {
            "tick_relative": 0,
            "addr_axon": 0,
            "addr_core_xy": 0,
            "addr_core_x": 0,
            "addr_core_y": 0,
            "addr_copy_xy": 0,
            "addr_copy_x": 0,
            "addr_copy_y": 0,
        }

    @pytest.mark.parametrize(
        "params_update",
        [
            {},
            {
                "tick_relative": NeuronLim.TICK_RELATIVE_MAX,
                "addr_axon": NeuronLim.ADDR_AXON_MAX,
                "addr_core_xy": NeuronLim.ADDR_CORE_OFFSET_MAX,
                "addr_core_x": NeuronLim.ADDR_CORE_OFFSET_MAX,
                "addr_core_y": NeuronLim.ADDR_CORE_OFFSET_MAX,
                "addr_copy_xy": NeuronLim.ADDR_CORE_OFFSET_MAX,
                "addr_copy_x": NeuronLim.ADDR_CORE_OFFSET_MAX,
                "addr_copy_y": NeuronLim.ADDR_CORE_OFFSET_MAX,
            },
        ],
    )
    def test_legal(self, ensure_dump_dir, default_params, params_update):
        params = default_params.copy()
        params.update(params_update)
        neuron = NeuronDestination2_5.model_validate(params, strict=True)
        neuron_dict = neuron.model_dump_json(indent=2)

        with open(ensure_dump_dir / "neuron_destination2_5.json", "w") as f:
            f.write(neuron_dict)

    @pytest.mark.parametrize(
        "params_update",
        [
            {"tick_relative": NeuronLim.TICK_RELATIVE_MAX + 1},
            {"addr_axon": NeuronLim.ADDR_AXON_MAX + 1},
            {"addr_core_xy": NeuronLim.ADDR_CORE_OFFSET_MIN - 1},
            {"addr_core_xy": NeuronLim.ADDR_CORE_OFFSET_MAX + 1},
        ],
    )
    def test_illegal(self, default_params, params_update):
        params = default_params.copy()
        params.update(params_update)
        with pytest.raises(ValidationError):
            NeuronDestination2_5.model_validate(params, strict=True)


class TestNeuronDifferentModel:
    @pytest.fixture
    def default_params(self):
        return {
            "weight_skew": 0,
            "weight_address_start": 0,
            "weight_address_end": 0,
            "output_type": OutputType.VALUE,
            "fold_type": FoldType.UNFOLDED,
            "neuron_type": NeuronType.HALF,
            "vjt": 0,
        }

    @pytest.mark.parametrize(
        "params_update",
        [
            {},
            {
                "weight_skew": NeuronLim.WEIGHT_SKEW_MAX,
                "weight_address_start": NeuronLim.WEIGHT_ADDRESS_MAX,
                "weight_address_end": NeuronLim.WEIGHT_ADDRESS_MAX,
                "output_type": OutputType.POTENTIAL,
                "fold_type": FoldType.FOLDED,
                "neuron_type": NeuronType.FULL,
                "vjt": 100,
            },
        ],
    )
    def test_legal(self, ensure_dump_dir, default_params, params_update):
        params = default_params.copy()
        params.update(params_update)
        neuron = NeuronDifferent.model_validate(params, strict=True)
        neuron_dict = neuron.model_dump_json(indent=2)

        with open(ensure_dump_dir / "neuron_different.json", "w") as f:
            f.write(neuron_dict)

    @pytest.mark.parametrize(
        "params_update",
        [
            {"weight_skew": NeuronLim.WEIGHT_SKEW_MAX + 1},
            {"weight_address_start": NeuronLim.WEIGHT_ADDRESS_MAX + 1},
        ],
    )
    def test_illegal(self, default_params, params_update):
        params = default_params.copy()
        params.update(params_update)
        with pytest.raises(ValidationError):
            NeuronDifferent.model_validate(params, strict=True)


class TestNeuronCommonModel:
    @pytest.fixture
    def default_params(self):
        return {
            "reset_mode": ResetMode.MODE_NORMAL,
            "reset_v": 0,
            "threshold_neg_mode": ThresholdNegMode.FIRE,
            "threshold_pos_mode": ThresholdPosMode.FIRE,
            "threshold_neg": 0,
            "threshold_pos": 0,
            "lateral_inhibition": LateralInhibitionMode.DISABLE,
            "leakmulti_sequence": LeakMultiComparisonOrder.BEFORE_COMPARE,
            "leakmulti_input": LeakMultiInputMode.DISABLE,
            "leakmulti_mode": LeakMultiMode.DISABLE,
            "leak_add_mode": LeakAddMode.FORWARD,
            "leak_tau": 0,
            "leak_v": 0,
            "weight_compress": WeightCompressMode.DENSE,
            "vjt_initial": 0,
        }

    @pytest.mark.parametrize(
        "params_update",
        [
            {},
            {
                "reset_mode": ResetMode.MODE_LINEAR,
                "reset_v": NeuronLim.RESET_V_MAX,
                "threshold_neg_mode": ThresholdNegMode.FLOOR,
                "threshold_pos_mode": ThresholdPosMode.CEILING,
                "threshold_neg": -100,
                "threshold_pos": 100,
                "lateral_inhibition": LateralInhibitionMode.ENABLE,
                "leakmulti_sequence": LeakMultiComparisonOrder.AFTER_COMPARE,
                "leakmulti_input": LeakMultiInputMode.ENABLE,
                "leakmulti_mode": LeakMultiMode.ENABLE,
                "leak_add_mode": LeakAddMode.BACKWARD,
                "leak_tau": NeuronLim.LEAK_TAU_MAX,
                "leak_v": NeuronLim.LEAK_V_MAX,
                "weight_compress": WeightCompressMode.SPARSE,
                "vjt_initial": NeuronLim.VJT_INITIAL_MAX,
            },
        ],
    )
    def test_legal(self, ensure_dump_dir, default_params, params_update):
        params = default_params.copy()
        params.update(params_update)
        neuron = NeuronCommon.model_validate(params, strict=True)
        neuron_dict = neuron.model_dump_json(indent=2)

        with open(ensure_dump_dir / "neuron_common.json", "w") as f:
            f.write(neuron_dict)

    @pytest.mark.parametrize(
        "params_update",
        [
            {"reset_v": NeuronLim.RESET_V_MIN - 1},
            {"reset_v": NeuronLim.RESET_V_MAX + 1},
            {"leak_tau": NeuronLim.LEAK_TAU_MIN - 1},
            {"leak_tau": NeuronLim.LEAK_TAU_MAX + 1},
            {"leak_v": NeuronLim.LEAK_V_MIN - 1},
            {"leak_v": NeuronLim.LEAK_V_MAX + 1},
            {"vjt_initial": NeuronLim.VJT_INITIAL_MIN - 1},
            {"vjt_initial": NeuronLim.VJT_INITIAL_MAX + 1},
        ],
    )
    def test_illegal(self, default_params, params_update):
        params = default_params.copy()
        params.update(params_update)
        with pytest.raises(ValidationError):
            NeuronCommon.model_validate(params, strict=True)


class TestFoldedNeuronParameterModel:
    @pytest.fixture
    def default_params(self):
        return {
            "fold_range_xy": 1,
            "fold_range_x": 1,
            "fold_range_y": 1,
            "fold_skew_xy": 0,
            "fold_skew_x": 0,
            "fold_skew_y": 0,
            "fold_axon_xy": 0,
            "fold_axon_x": 0,
            "fold_axon_y": 0,
            "fold_number": 1,
        }

    @pytest.mark.parametrize(
        "params_update",
        [
            {},
            {
                "fold_range_xy": 2,
                "fold_range_x": 2,
                "fold_range_y": 2,
                "fold_number": 8,
            },
        ],
    )
    def test_legal(self, ensure_dump_dir, default_params, params_update):
        params = default_params.copy()
        params.update(params_update)
        neuron = FoldedNeuronParameter.model_validate(params, strict=True)
        neuron_dict = neuron.model_dump_json(indent=2)

        with open(ensure_dump_dir / "folded_neuron_parameter.json", "w") as f:
            f.write(neuron_dict)

    @pytest.mark.parametrize(
        "params_update",
        [
            {"fold_range_xy": NeuronLim.FOLD_RANGE_MAX + 1},
            {"fold_skew_xy": NeuronLim.FOLD_SKEW_MAX + 1},
            {"fold_axon_xy": NeuronLim.FOLD_AXON_MAX + 1},
            {"fold_number": NeuronLim.FOLD_NUMBER_MAX + 1},
            # Logic error: fold_number != range_xy * range_x * range_y
            {"fold_range_xy": 2, "fold_number": 1},
        ],
    )
    def test_illegal(self, default_params, params_update):
        params = default_params.copy()
        params.update(params_update)
        with pytest.raises((ValidationError, ValueError)):
            FoldedNeuronParameter.model_validate(params, strict=True)


class TestFoldedNeuronPotentialModel:
    @pytest.fixture
    def default_params(self):
        return {
            "fold_vjt_3": 0,
            "fold_vjt_2": 0,
            "fold_vjt_1": 0,
            "fold_vjt_0": 0,
        }

    @pytest.mark.parametrize(
        "params_update",
        [
            {},
            {
                "fold_vjt_3": 100,
                "fold_vjt_2": -100,
                "fold_vjt_1": 50,
                "fold_vjt_0": -50,
            },
        ],
    )
    def test_legal(self, ensure_dump_dir, default_params, params_update):
        params = default_params.copy()
        params.update(params_update)
        neuron = FoldedNeuronPotential.model_validate(params, strict=True)
        neuron_dict = neuron.model_dump_json(indent=2)

        with open(ensure_dump_dir / "folded_neuron_potential.json", "w") as f:
            f.write(neuron_dict)
