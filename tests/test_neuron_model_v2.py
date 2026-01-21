import pytest
from pydantic import ValidationError

from paicorelib.neuron_defs import ResetMode
from paicorelib.neuron_defs_v2 import (
    FoldType,
    LateralInhibitionMode,
    LeakAddMode,
    LeakMultiComparisonOrder,
    LeakMultiInputMode,
    LeakMultiMode,
    NeuronType,
    OfflineNeuRegLimV2,
    OutputType,
    ThresholdNegMode,
    ThresholdPosMode,
    WeightCompressType,
)
from paicorelib.neuron_model_v2 import (
    OfflineNeuDestInfoV2,
    OfflineNeuFoldedAttrsV2Part1,
    OfflineNeuFullAttrsV2,
    OfflineNeuHalfAttrsV2,
)


class TestOfflineNeuDestInfoV2Model:
    @pytest.fixture
    def default_params(self):
        return {
            "tick_relative": 1,
            "addr_axon": 1,
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
                "tick_relative": 0,
                "addr_axon": 99,
                "addr_core_xy": OfflineNeuRegLimV2.ADDR_CORE_COORD_MAX,
                "addr_core_x": OfflineNeuRegLimV2.ADDR_CORE_COORD_MAX,
                "addr_core_y": OfflineNeuRegLimV2.ADDR_CORE_COORD_MAX,
                "addr_copy_xy": OfflineNeuRegLimV2.ADDR_CORE_COORD_MAX,
                "addr_copy_x": OfflineNeuRegLimV2.ADDR_CORE_COORD_MAX,
                "addr_copy_y": OfflineNeuRegLimV2.ADDR_CORE_COORD_MAX,
            },
        ],
    )
    def test_legal(self, ensure_dump_dir, default_params, params_update):
        params = default_params.copy()
        params.update(params_update)
        neuron = OfflineNeuDestInfoV2.model_validate(params, strict=True)
        neuron_dict = neuron.model_dump_json(indent=2)

        with open(ensure_dump_dir / "neuron_destination2_5.json", "w") as f:
            f.write(neuron_dict)

    @pytest.mark.parametrize(
        "params_update",
        [
            {"tick_relative": OfflineNeuRegLimV2.TICK_RELATIVE_MAX + 1},
            {"addr_axon": OfflineNeuRegLimV2.ADDR_AXON_MAX + 1},
            {"addr_core_xy": OfflineNeuRegLimV2.ADDR_CORE_COORD_MIN - 1},
            {"addr_core_xy": OfflineNeuRegLimV2.ADDR_CORE_COORD_MAX + 1},
        ],
    )
    def test_illegal(self, default_params, params_update):
        params = default_params.copy()
        params.update(params_update)
        with pytest.raises(ValidationError):
            OfflineNeuDestInfoV2.model_validate(params, strict=True)


class TestOfflineNeuHalfAttrsV2Model:
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
                "weight_skew": OfflineNeuRegLimV2.WEIGHT_SKEW_MAX,
                "weight_address_start": OfflineNeuRegLimV2.WEIGHT_ADDRESS_MAX,
                "weight_address_end": OfflineNeuRegLimV2.WEIGHT_ADDRESS_MAX,
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
        neuron = OfflineNeuHalfAttrsV2.model_validate(params, strict=True)
        neuron_dict = neuron.model_dump_json(indent=2)

        with open(ensure_dump_dir / "neuron_half_attrs.json", "w") as f:
            f.write(neuron_dict)

    @pytest.mark.parametrize(
        "params_update",
        [
            {"weight_skew": OfflineNeuRegLimV2.WEIGHT_SKEW_MAX + 1},
            {"weight_address_start": OfflineNeuRegLimV2.WEIGHT_ADDRESS_MAX + 1},
        ],
    )
    def test_illegal(self, default_params, params_update):
        params = default_params.copy()
        params.update(params_update)
        with pytest.raises(ValidationError):
            OfflineNeuHalfAttrsV2.model_validate(params, strict=True)


class TestOfflineNeuFullAttrsV2Model:
    @pytest.fixture
    def default_params(self):
        return {
            "weight_skew": 0,
            "weight_address_start": 0,
            "weight_address_end": 1000,
            "output_type": OutputType.VALUE,
            "fold_type": FoldType.UNFOLDED,
            "neuron_type": NeuronType.FULL,
            "vjt": 0,
            "reset_mode": ResetMode.MODE_NORMAL,
            "reset_v": 0,
            "threshold_neg_mode": ThresholdNegMode.FIRE,
            "threshold_pos_mode": ThresholdPosMode.FIRE,
            "threshold_neg": 0,
            "threshold_pos": 0,
            "lateral_inhibition": LateralInhibitionMode.DISABLE,
            "leak_multi_sequence": LeakMultiComparisonOrder.BEFORE_COMPARE,
            "leak_multi_input": LeakMultiInputMode.DISABLE,
            "leak_multi_mode": LeakMultiMode.DISABLE,
            "leak_add_mode": LeakAddMode.FORWARD,
            "leak_tau": 0,
            "leak_v": 0,
            "weight_compress": WeightCompressType.DENSE,
            "vjt_initial": 0,
        }

    @pytest.mark.parametrize(
        "params_update",
        [
            {},
            {
                "weight_skew": 1,
                "weight_address_start": 0,
                "weight_address_end": 1,
                "output_type": OutputType.POTENTIAL,
                "fold_type": FoldType.FOLDED,
                "neuron_type": NeuronType.FULL,
                "vjt": 0,
                "reset_mode": ResetMode.MODE_LINEAR,
                "reset_v": OfflineNeuRegLimV2.RESET_V_MAX,
                "threshold_neg_mode": ThresholdNegMode.FLOOR,
                "threshold_pos_mode": ThresholdPosMode.CEILING,
                "threshold_neg": -100,
                "threshold_pos": 100,
                "lateral_inhibition": LateralInhibitionMode.ENABLE,
                "leak_multi_sequence": LeakMultiComparisonOrder.AFTER_COMPARE,
                "leak_multi_input": LeakMultiInputMode.ENABLE,
                "leak_multi_mode": LeakMultiMode.ENABLE,
                "leak_add_mode": LeakAddMode.BACKWARD,
                "leak_tau": OfflineNeuRegLimV2.LEAK_TAU_MAX,
                "leak_v": OfflineNeuRegLimV2.LEAK_V_MAX,
                "weight_compress": WeightCompressType.SPARSE,
                "vjt_initial": OfflineNeuRegLimV2.VJT_INITIAL_MAX,
            },
        ],
    )
    def test_legal(self, ensure_dump_dir, default_params, params_update):
        params = default_params.copy()
        params.update(params_update)
        neuron = OfflineNeuFullAttrsV2.model_validate(params, strict=True)
        neuron_dict = neuron.model_dump_json(indent=2)

        with open(ensure_dump_dir / "neuron_common.json", "w") as f:
            f.write(neuron_dict)

    @pytest.mark.parametrize(
        "params_update",
        [
            {"reset_v": OfflineNeuRegLimV2.RESET_V_MIN - 1},
            {"reset_v": OfflineNeuRegLimV2.RESET_V_MAX + 1},
            {"leak_tau": OfflineNeuRegLimV2.LEAK_TAU_MIN - 1},
            {"leak_tau": OfflineNeuRegLimV2.LEAK_TAU_MAX + 1},
            {"leak_v": OfflineNeuRegLimV2.LEAK_V_MIN - 1},
            {"leak_v": OfflineNeuRegLimV2.LEAK_V_MAX + 1},
            {"vjt_initial": OfflineNeuRegLimV2.VJT_INITIAL_MIN - 1},
            {"vjt_initial": OfflineNeuRegLimV2.VJT_INITIAL_MAX + 1},
        ],
    )
    def test_illegal(self, default_params, params_update):
        params = default_params.copy()
        params.update(params_update)
        with pytest.raises(ValidationError):
            OfflineNeuFullAttrsV2.model_validate(params, strict=True)


class TestOfflineNeuFoldedAttrsV2Part1Model:
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
        neuron = OfflineNeuFoldedAttrsV2Part1.model_validate(params, strict=True)
        neuron_dict = neuron.model_dump_json(indent=2)

        with open(ensure_dump_dir / "folded_neuron_attrs.json", "w") as f:
            f.write(neuron_dict)

    @pytest.mark.parametrize(
        "params_update",
        [
            {"fold_range_xy": OfflineNeuRegLimV2.FOLD_RANGE_MAX + 1},
            {"fold_skew_xy": OfflineNeuRegLimV2.FOLD_SKEW_MAX + 1},
            {"fold_axon_xy": OfflineNeuRegLimV2.FOLD_AXON_MAX + 1},
            {"fold_number": OfflineNeuRegLimV2.FOLD_NUMBER_MAX + 1},
            # Logic error: fold_number != range_xy * range_x * range_y
            {"fold_range_xy": 2, "fold_number": 1},
        ],
    )
    def test_illegal(self, default_params, params_update):
        params = default_params.copy()
        params.update(params_update)
        with pytest.raises((ValidationError, ValueError)):
            OfflineNeuFoldedAttrsV2Part1.model_validate(params, strict=True)
