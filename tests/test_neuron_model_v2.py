import pytest
from pydantic import ValidationError

from paicorelib.neuron_defs_v2 import (
    FoldType,
    NeuronType,
    OfflineNeuRegLimV2,
    OutputType,
)
from paicorelib.neuron_model_v2 import (
    OfflineNeuDestInfoV2,
    OfflineNeuFoldedAttrsV2Part1,
    OfflineNeuFoldedAttrsV2Part2,
    OfflineNeuFullAttrsV2,
    OfflineNeuFullConfV2,
    OfflineNeuHalfAttrsV2,
    OfflineNeuHalfConfV2,
    OnlineNeuFoldedAttrsV2Part1,
    OnlineNeuFoldedAttrsV2Part2,
)
from tests.utils import (
    build_v2_dest_info_params,
    build_v2_folded_attrs_part1_params,
    build_v2_folded_attrs_part2_params,
    build_v2_full_attrs_part2_params,
    build_v2_half_attrs_params,
)


class TestOfflineNeuDestInfoV2Model:
    @pytest.mark.parametrize(
        "params",
        [
            build_v2_dest_info_params(),
            build_v2_dest_info_params(
                tick_relative=0,
                addr_axon=99,
                addr_core_xy=OfflineNeuRegLimV2.ADDR_CORE_COORD_MAX,
                addr_core_x=OfflineNeuRegLimV2.ADDR_CORE_COORD_MAX,
                addr_core_y=OfflineNeuRegLimV2.ADDR_CORE_COORD_MAX,
                addr_copy_xy=OfflineNeuRegLimV2.ADDR_CORE_COORD_MAX,
                addr_copy_x=OfflineNeuRegLimV2.ADDR_CORE_COORD_MAX,
                addr_copy_y=OfflineNeuRegLimV2.ADDR_CORE_COORD_MAX,
            ),
        ],
    )
    def test_legal(self, ensure_dump_dir, params):
        neuron = OfflineNeuDestInfoV2.model_validate(params, strict=True)

        with open(ensure_dump_dir / "neuron_destination2_5.json", "w") as f:
            f.write(neuron.model_dump_json(indent=2))

    @pytest.mark.parametrize(
        "params_update",
        [
            {"tick_relative": OfflineNeuRegLimV2.TICK_RELATIVE_MAX + 1},
            {"addr_axon": OfflineNeuRegLimV2.ADDR_AXON_MAX + 1},
            {"addr_core_xy": OfflineNeuRegLimV2.ADDR_CORE_COORD_MIN - 1},
            {"addr_core_xy": OfflineNeuRegLimV2.ADDR_CORE_COORD_MAX + 1},
        ],
    )
    def test_illegal(self, params_update):
        with pytest.raises(ValidationError):
            OfflineNeuDestInfoV2.model_validate(
                build_v2_dest_info_params(**params_update), strict=True
            )


class TestOfflineNeuHalfAttrsV2Model:
    @pytest.mark.parametrize(
        "params",
        [
            build_v2_half_attrs_params(),
            build_v2_half_attrs_params(
                weight_skew=OfflineNeuRegLimV2.WEIGHT_SKEW_MAX,
                weight_address_start=OfflineNeuRegLimV2.WEIGHT_ADDRESS_MAX,
                weight_address_end=OfflineNeuRegLimV2.WEIGHT_ADDRESS_MAX,
                output_type=OutputType.POTENTIAL,
                fold_type=FoldType.FOLDED,
                neuron_type=NeuronType.FULL,
                vjt=100,
            ),
        ],
    )
    def test_legal(self, ensure_dump_dir, params):
        neuron = OfflineNeuHalfAttrsV2.model_validate(params, strict=True)

        with open(ensure_dump_dir / "neuron_half_attrs.json", "w") as f:
            f.write(neuron.model_dump_json(indent=2))

    @pytest.mark.parametrize(
        "params_update",
        [
            {"weight_skew": OfflineNeuRegLimV2.WEIGHT_SKEW_MAX + 1},
            {"weight_address_start": OfflineNeuRegLimV2.WEIGHT_ADDRESS_MAX + 1},
        ],
    )
    def test_illegal(self, params_update):
        with pytest.raises(ValidationError):
            OfflineNeuHalfAttrsV2.model_validate(
                build_v2_half_attrs_params(**params_update), strict=True
            )


class TestOfflineNeuFullAttrsV2Model:
    @pytest.mark.parametrize(
        "common_update, part2_update",
        [
            ({}, {}),
            (
                {
                    "weight_skew": 1,
                    "weight_address_end": 1,
                    "output_type": OutputType.POTENTIAL,
                    "fold_type": FoldType.FOLDED,
                    "neuron_type": NeuronType.FULL,
                },
                {
                    "reset_v": OfflineNeuRegLimV2.RESET_V_MAX,
                    "threshold_neg": -100,
                    "threshold_pos": 100,
                    "leak_tau": OfflineNeuRegLimV2.LEAK_TAU_MAX,
                    "leak_v": OfflineNeuRegLimV2.LEAK_V_MAX,
                    "vjt_initial": OfflineNeuRegLimV2.VJT_INITIAL_MAX,
                },
            ),
        ],
    )
    def test_legal(self, ensure_dump_dir, common_update, part2_update):
        params = build_v2_half_attrs_params(
            **({"neuron_type": NeuronType.FULL} | common_update)
        )
        params.update(build_v2_full_attrs_part2_params(**part2_update))
        neuron = OfflineNeuFullAttrsV2.model_validate(params, strict=True)

        with open(ensure_dump_dir / "neuron_common.json", "w") as f:
            f.write(neuron.model_dump_json(indent=2))

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
    def test_illegal(self, params_update):
        params = build_v2_half_attrs_params(neuron_type=NeuronType.FULL)
        params.update(build_v2_full_attrs_part2_params(**params_update))

        with pytest.raises(ValidationError):
            OfflineNeuFullAttrsV2.model_validate(params, strict=True)


class TestOfflineNeuFoldedAttrsV2Part1Model:
    @pytest.mark.parametrize(
        "params",
        [
            build_v2_folded_attrs_part1_params(),
            build_v2_folded_attrs_part1_params(
                fold_range_xy=2,
                fold_range_x=2,
                fold_range_y=2,
                fold_number=8,
            ),
        ],
    )
    def test_legal(self, ensure_dump_dir, params):
        neuron = OfflineNeuFoldedAttrsV2Part1.model_validate(params, strict=True)

        with open(ensure_dump_dir / "folded_neuron_attrs.json", "w") as f:
            f.write(neuron.model_dump_json(indent=2))

    @pytest.mark.parametrize(
        "params_update",
        [
            {"fold_range_xy": OfflineNeuRegLimV2.FOLD_RANGE_MAX + 1},
            {"fold_skew_xy": OfflineNeuRegLimV2.FOLD_SKEW_MAX + 1},
            {"fold_axon_xy": OfflineNeuRegLimV2.FOLD_AXON_MAX + 1},
            {"fold_number": OfflineNeuRegLimV2.FOLD_NUMBER_MAX + 1},
            {"fold_range_xy": 2, "fold_number": 1},
        ],
    )
    def test_illegal(self, params_update):
        with pytest.raises((ValidationError, ValueError)):
            OfflineNeuFoldedAttrsV2Part1.model_validate(
                build_v2_folded_attrs_part1_params(**params_update), strict=True
            )


class TestAdditionalV2Models:
    def test_offline_folded_attrs_part2(self):
        folded_attrs = OfflineNeuFoldedAttrsV2Part2.model_validate(
            build_v2_folded_attrs_part2_params(
                fold_vjt_3=300,
                fold_vjt_2=200,
                fold_vjt_1=100,
                fold_vjt_0=0,
            ),
            strict=True,
        )

        assert folded_attrs.fold_vjt_3 == 300
        assert folded_attrs.fold_vjt_0 == 0

    def test_online_folded_attrs_models_accept_float_values(self):
        attrs1 = OnlineNeuFoldedAttrsV2Part1.model_validate(
            build_v2_folded_attrs_part1_params(), strict=True
        )
        attrs2 = OnlineNeuFoldedAttrsV2Part2.model_validate(
            {
                "fold_vjt_3": 3.5,
                "fold_vjt_2": 2.5,
                "fold_vjt_1": 1.5,
                "fold_vjt_0": 0.5,
            },
            strict=True,
        )

        assert attrs1.fold_number == 1
        assert attrs2.fold_vjt_3 == 3.5

    def test_offline_conf_wrappers_validate_nested_models(self):
        half_conf = OfflineNeuHalfConfV2.model_validate(
            {
                "attrs": build_v2_half_attrs_params(),
                "dest_info": build_v2_dest_info_params(),
            },
            strict=True,
        )
        full_conf = OfflineNeuFullConfV2.model_validate(
            {
                "attrs": {
                    **build_v2_half_attrs_params(neuron_type=NeuronType.FULL),
                    **build_v2_full_attrs_part2_params(),
                },
                "dest_info": build_v2_dest_info_params(),
            },
            strict=True,
        )

        assert isinstance(half_conf.attrs, OfflineNeuHalfAttrsV2)
        assert isinstance(half_conf.dest_info, OfflineNeuDestInfoV2)
        assert isinstance(full_conf.attrs, OfflineNeuFullAttrsV2)
