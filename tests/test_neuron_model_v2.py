import numpy as np
import pytest
from pydantic import ValidationError

from paicorelib.float_codec import cast_fp32_scalar
from paicorelib.neuron_defs_v2 import (
    FoldType,
    NeuronType,
    OfflineNeuRegLimV2,
    OnlineNeuRegLimV2,
    OnlineOutputType,
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
    OnlineNeuDestInfoV2,
    OnlineNeuFoldedAttrsV2Part1,
    OnlineNeuFoldedAttrsV2Part2,
    OnlineNeuFullAttrsV2,
    OnlineNeuFullConfV2,
    OnlineNeuHalfAttrsV2,
    OnlineNeuHalfConfV2,
)
from tests.utils import (
    build_online_v2_half_attrs_params,
    build_v2_dest_info_params,
    build_v2_folded_attrs_part1_params,
    build_v2_folded_attrs_part2_params,
    build_v2_full_attrs_part2_params,
    build_v2_half_attrs_params,
)


class TestOfflineNeuParamModelV2:
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
    def test_dest_info_legal(self, ensure_dump_dir, params):
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
    def test_dest_info_illegal(self, params_update):
        with pytest.raises(ValidationError):
            OfflineNeuDestInfoV2.model_validate(
                build_v2_dest_info_params(**params_update), strict=True
            )

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
    def test_half_attrs_legal(self, ensure_dump_dir, params):
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
    def test_half_attrs_illegal(self, params_update):
        with pytest.raises(ValidationError):
            OfflineNeuHalfAttrsV2.model_validate(
                build_v2_half_attrs_params(**params_update), strict=True
            )

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
    def test_full_attrs_legal(self, ensure_dump_dir, common_update, part2_update):
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
    def test_full_attrs_illegal(self, params_update):
        params = build_v2_half_attrs_params(neuron_type=NeuronType.FULL)
        params.update(build_v2_full_attrs_part2_params(**params_update))

        with pytest.raises(ValidationError):
            OfflineNeuFullAttrsV2.model_validate(params, strict=True)

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
    def test_folded_attrs_part1_legal(self, ensure_dump_dir, params):
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
    def test_folded_attrs_part1_illegal(self, params_update):
        with pytest.raises((ValidationError, ValueError)):
            OfflineNeuFoldedAttrsV2Part1.model_validate(
                build_v2_folded_attrs_part1_params(**params_update), strict=True
            )

    def test_folded_attrs_part2_legal(self):
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

    def test_conf_wrappers_validate_nested_models(self):
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


class TestOnlineNeuParamModelV2:
    @pytest.mark.parametrize(
        "params",
        [
            build_v2_dest_info_params(),
            build_v2_dest_info_params(addr_axon=OnlineNeuRegLimV2.ADDR_AXON_MAX),
        ],
    )
    def test_dest_info_legal(self, params):
        dest_info = OnlineNeuDestInfoV2.model_validate(params, strict=True)
        assert dest_info.addr_axon == params["addr_axon"]

    @pytest.mark.parametrize(
        "params_update",
        [
            {"addr_axon": OnlineNeuRegLimV2.ADDR_AXON_MAX + 1},
            {"addr_core_xy": OfflineNeuRegLimV2.ADDR_CORE_COORD_MIN - 1},
        ],
    )
    def test_dest_info_illegal(self, params_update):
        with pytest.raises(ValidationError):
            OnlineNeuDestInfoV2.model_validate(
                build_v2_dest_info_params(**params_update), strict=True
            )

    @pytest.mark.parametrize(
        "params",
        [
            build_online_v2_half_attrs_params(vjt=np.float32(0.0)),
            build_online_v2_half_attrs_params(
                weight_skew=OfflineNeuRegLimV2.WEIGHT_SKEW_MAX,
                weight_address_start=OfflineNeuRegLimV2.WEIGHT_ADDRESS_MAX,
                weight_address_end=OfflineNeuRegLimV2.WEIGHT_ADDRESS_MAX,
                output_type=OnlineOutputType.VALUE_AND_POTENTIAL_16BIT,
                fold_type=FoldType.FOLDED,
                neuron_type=NeuronType.FULL,
                vjt=np.float64(1.0 / 3.0),
            ),
        ],
    )
    def test_half_attrs_legal(self, params):
        half_attrs = OnlineNeuHalfAttrsV2.model_validate(params, strict=True)
        assert half_attrs.output_type == params["output_type"]
        assert half_attrs.vjt == cast_fp32_scalar(params["vjt"])

    @pytest.mark.parametrize(
        "params_update",
        [
            {"weight_skew": OfflineNeuRegLimV2.WEIGHT_SKEW_MAX + 1},
            {"weight_address_start": OfflineNeuRegLimV2.WEIGHT_ADDRESS_MAX + 1},
            {"output_type": OutputType.VALUE},
            {"output_type": 1},
        ],
    )
    def test_half_attrs_illegal(self, params_update):
        with pytest.raises(ValidationError):
            OnlineNeuHalfAttrsV2.model_validate(
                build_online_v2_half_attrs_params(**params_update), strict=True
            )

    @pytest.mark.parametrize(
        "common_update, part2_update",
        [
            ({}, {}),
            (
                {
                    "weight_skew": 1,
                    "weight_address_end": 1,
                    "output_type": OnlineOutputType.VALUE_AND_MAX_POOLING_POSITION,
                    "fold_type": FoldType.FOLDED,
                    "neuron_type": NeuronType.FULL,
                    "vjt": np.float32(-0.1),
                },
                {
                    "reset_v": np.float32(-0.75),
                    "threshold_neg": np.float64(-2.5),
                    "threshold_pos": np.float32(3.25),
                    "leak_tau": OnlineNeuRegLimV2.LEAK_TAU_MAX,
                    "leak_v": np.float64(-1.5),
                    "vjt_initial": np.float32(0.5),
                },
            ),
        ],
    )
    def test_full_attrs_legal(self, common_update, part2_update):
        params = build_online_v2_half_attrs_params(
            **({"neuron_type": NeuronType.FULL, "vjt": np.float32(0.0)} | common_update)
        )
        params.update(build_v2_full_attrs_part2_params(**part2_update))
        full_attrs = OnlineNeuFullAttrsV2.model_validate(params, strict=True)

        assert full_attrs.vjt == cast_fp32_scalar(params["vjt"])
        assert full_attrs.reset_v == cast_fp32_scalar(params["reset_v"])
        assert full_attrs.threshold_neg == cast_fp32_scalar(params["threshold_neg"])
        assert full_attrs.threshold_pos == cast_fp32_scalar(params["threshold_pos"])
        assert full_attrs.leak_v == cast_fp32_scalar(params["leak_v"])
        assert full_attrs.vjt_initial == cast_fp32_scalar(params["vjt_initial"])

    @pytest.mark.parametrize(
        "params_update",
        [
            {"leak_tau": OnlineNeuRegLimV2.LEAK_TAU_MIN - 1},
            {"leak_tau": OnlineNeuRegLimV2.LEAK_TAU_MAX + 1},
        ],
    )
    def test_full_attrs_illegal(self, params_update):
        params = build_online_v2_half_attrs_params(neuron_type=NeuronType.FULL)
        params.update(build_v2_full_attrs_part2_params(**params_update))

        with pytest.raises(ValidationError):
            OnlineNeuFullAttrsV2.model_validate(params, strict=True)

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
    def test_folded_attrs_part1_legal(self, params):
        attrs = OnlineNeuFoldedAttrsV2Part1.model_validate(params, strict=True)

        assert attrs.fold_number == params["fold_number"]

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
    def test_folded_attrs_part1_illegal(self, params_update):
        with pytest.raises((ValidationError, ValueError)):
            OnlineNeuFoldedAttrsV2Part1.model_validate(
                build_v2_folded_attrs_part1_params(**params_update), strict=True
            )

    def test_folded_attrs_part2_legal(self):
        folded_attrs = OnlineNeuFoldedAttrsV2Part2.model_validate(
            build_v2_folded_attrs_part2_params(
                fold_vjt_3=np.float32(1.0),
                fold_vjt_2=np.float64(-2.5),
                fold_vjt_1=np.float32(3.25),
                fold_vjt_0=np.float64(-4.75),
            ),
            strict=True,
        )

        assert folded_attrs.fold_vjt_3 == cast_fp32_scalar(np.float32(1.0))
        assert folded_attrs.fold_vjt_2 == cast_fp32_scalar(np.float64(-2.5))
        assert folded_attrs.fold_vjt_1 == cast_fp32_scalar(np.float32(3.25))
        assert folded_attrs.fold_vjt_0 == cast_fp32_scalar(np.float64(-4.75))

    def test_conf_wrappers_validate_nested_models(self):
        half_conf = OnlineNeuHalfConfV2.model_validate(
            {
                "attrs": build_online_v2_half_attrs_params(
                    output_type=OnlineOutputType.VALUE_AND_POTENTIAL_16BIT, vjt=0.5
                ),
                "dest_info": build_v2_dest_info_params(),
            },
            strict=True,
        )
        full_conf = OnlineNeuFullConfV2.model_validate(
            {
                "attrs": {
                    **build_online_v2_half_attrs_params(
                        neuron_type=NeuronType.FULL,
                        output_type=OnlineOutputType.VALUE_AND_MAX_POOLING_POSITION,
                        vjt=0.5,
                    ),
                    **build_v2_full_attrs_part2_params(
                        reset_v=0.25,
                        threshold_neg=-1.0,
                        threshold_pos=1.0,
                        leak_v=0.125,
                        vjt_initial=0.0,
                    ),
                },
                "dest_info": build_v2_dest_info_params(),
            },
            strict=True,
        )

        assert isinstance(half_conf.attrs, OnlineNeuHalfAttrsV2)
        assert isinstance(half_conf.dest_info, OnlineNeuDestInfoV2)
        assert half_conf.attrs.output_type == OnlineOutputType.VALUE_AND_POTENTIAL_16BIT
        assert isinstance(full_conf.attrs, OnlineNeuFullAttrsV2)
        assert isinstance(full_conf.dest_info, OnlineNeuDestInfoV2)
        assert (
            full_conf.attrs.output_type
            == OnlineOutputType.VALUE_AND_MAX_POOLING_POSITION
        )
