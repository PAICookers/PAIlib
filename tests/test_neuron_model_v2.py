from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from paicorelib import neuron_model_v2 as neuron_models_v2
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

NeuDestInfoModel = type[OfflineNeuDestInfoV2] | type[OnlineNeuDestInfoV2]
NeuHalfAttrsModel = type[OfflineNeuHalfAttrsV2] | type[OnlineNeuHalfAttrsV2]
NeuFoldedAttrsPart1Model = (
    type[OfflineNeuFoldedAttrsV2Part1] | type[OnlineNeuFoldedAttrsV2Part1]
)
NeuFoldedAttrsPart2Model = (
    type[OfflineNeuFoldedAttrsV2Part2] | type[OnlineNeuFoldedAttrsV2Part2]
)

DEST_INFO_FIELDS = {
    "tick_relative",
    "addr_axon",
    "addr_core_xy",
    "addr_core_x",
    "addr_core_y",
    "addr_copy_xy",
    "addr_copy_x",
    "addr_copy_y",
}

COMMON_HALF_ATTR_FIELDS = {
    "weight_skew",
    "weight_address_start",
    "weight_address_end",
    "fold_type",
    "neuron_type",
}

PROTOCOL_SPECIFIC_HALF_ATTR_FIELDS = {"output_type", "vjt"}

FOLDED_PART1_FIELDS = {
    "fold_range_xy",
    "fold_range_x",
    "fold_range_y",
    "fold_skew_xy",
    "fold_skew_x",
    "fold_skew_y",
    "fold_axon_xy",
    "fold_axon_x",
    "fold_axon_y",
    "fold_number",
}

FULL_PART2_SHARED_CONTROL_FIELDS = {
    "reset_mode",
    "threshold_neg_mode",
    "threshold_pos_mode",
    "lateral_inhibition",
    "leak_multi_sequence",
    "leak_multi_input",
    "leak_multi_mode",
    "leak_add_mode",
    "leak_tau",
    "weight_compress",
}

FULL_PART2_PROTOCOL_NUMERIC_FIELDS = {
    "reset_v",
    "threshold_neg",
    "threshold_pos",
    "leak_v",
    "vjt_initial",
}

DEST_INFO_CASES = (
    pytest.param(OfflineNeuDestInfoV2, OfflineNeuRegLimV2.ADDR_AXON_MAX, id="offline"),
    pytest.param(OnlineNeuDestInfoV2, OnlineNeuRegLimV2.ADDR_AXON_MAX, id="online"),
)

HALF_ATTR_MODEL_CASES = (
    pytest.param(OfflineNeuHalfAttrsV2, build_v2_half_attrs_params, id="offline"),
    pytest.param(OnlineNeuHalfAttrsV2, build_online_v2_half_attrs_params, id="online"),
)

FOLDED_PART1_CASES = (
    pytest.param(OfflineNeuFoldedAttrsV2Part1, id="offline"),
    pytest.param(OnlineNeuFoldedAttrsV2Part1, id="online"),
)


def build_offline_full_attrs_params(
    part1_update: dict[str, Any] | None = None,
    part2_update: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = build_v2_half_attrs_params(
        neuron_type=NeuronType.FULL, **(part1_update or {})
    )
    params.update(build_v2_full_attrs_part2_params(**(part2_update or {})))
    return params


def build_online_full_attrs_params(
    part1_update: dict[str, Any] | None = None,
    part2_update: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = build_online_v2_half_attrs_params(
        neuron_type=NeuronType.FULL, **(part1_update or {})
    )
    params.update(build_v2_full_attrs_part2_params(**(part2_update or {})))
    return params


class TestNeuronModelV2FieldLayout:
    def test_common_attr_fields_live_on_shared_bases(self):
        assert set(OfflineNeuDestInfoV2.model_fields) == DEST_INFO_FIELDS
        assert set(OnlineNeuDestInfoV2.model_fields) == DEST_INFO_FIELDS
        assert set(neuron_models_v2._NeuCommonAttrsV2.model_fields) == (
            COMMON_HALF_ATTR_FIELDS
        )
        assert set(neuron_models_v2._NeuFoldedAttrsV2Part1.model_fields) == (
            FOLDED_PART1_FIELDS
        )
        assert PROTOCOL_SPECIFIC_HALF_ATTR_FIELDS.isdisjoint(
            neuron_models_v2._NeuCommonAttrsV2.model_fields
        )

    def test_protocol_specific_half_attr_types_stay_separate(self):
        assert (
            OfflineNeuHalfAttrsV2.model_fields["output_type"].annotation is OutputType
        )
        assert (
            OnlineNeuHalfAttrsV2.model_fields["output_type"].annotation
            is OnlineOutputType
        )
        assert OfflineNeuHalfAttrsV2.model_fields["vjt"].annotation is int
        assert OnlineNeuHalfAttrsV2.model_fields["vjt"].annotation is float

    def test_full_part2_only_control_fields_have_shared_types(self):
        for field in FULL_PART2_SHARED_CONTROL_FIELDS:
            assert (
                OfflineNeuFullAttrsV2.model_fields[field].annotation
                == OnlineNeuFullAttrsV2.model_fields[field].annotation
            )

        for field in FULL_PART2_PROTOCOL_NUMERIC_FIELDS:
            assert (
                OfflineNeuFullAttrsV2.model_fields[field].annotation
                != OnlineNeuFullAttrsV2.model_fields[field].annotation
            )

    def test_renamed_online_output_member_keeps_compatibility_alias(self):
        assert (
            OnlineOutputType.VALUE_AND_MAX_POOLING_POSITION
            is OnlineOutputType.VALUE_AND_MAX_POOLING_POSITIONS
        )
        assert (
            OnlineOutputType["VALUE_AND_MAX_POOLING_POSITION"]
            is OnlineOutputType.VALUE_AND_MAX_POOLING_POSITIONS
        )
        assert "VALUE_AND_MAX_POOLING_POSITION" in OnlineOutputType.__members__


class TestNeuDestInfoV2:
    @pytest.mark.parametrize(("model", "axon_max"), DEST_INFO_CASES)
    @pytest.mark.parametrize(
        "coord",
        [
            pytest.param(OfflineNeuRegLimV2.ADDR_CORE_COORD_MIN, id="coord-min"),
            pytest.param(0, id="coord-zero"),
            pytest.param(OfflineNeuRegLimV2.ADDR_CORE_COORD_MAX, id="coord-max"),
        ],
    )
    def test_accepts_legal_values(
        self, model: NeuDestInfoModel, axon_max: int, coord: int
    ):
        dest_info = model.model_validate(
            build_v2_dest_info_params(
                tick_relative=OfflineNeuRegLimV2.TICK_RELATIVE_MAX,
                addr_axon=axon_max,
                addr_core_xy=coord,
                addr_core_x=coord,
                addr_core_y=coord,
                addr_copy_xy=coord,
                addr_copy_x=coord,
                addr_copy_y=coord,
            ),
            strict=True,
        )

        assert dest_info.addr_axon == axon_max
        assert dest_info.addr_core_xy == coord

    @pytest.mark.parametrize(("model", "axon_max"), DEST_INFO_CASES)
    @pytest.mark.parametrize(
        "params_update",
        [
            pytest.param(
                {"tick_relative": OfflineNeuRegLimV2.TICK_RELATIVE_MAX + 1},
                id="tick-relative-high",
            ),
            pytest.param(
                {"addr_core_xy": OfflineNeuRegLimV2.ADDR_CORE_COORD_MIN - 1},
                id="core-xy-low",
            ),
            pytest.param(
                {"addr_core_xy": OfflineNeuRegLimV2.ADDR_CORE_COORD_MAX + 1},
                id="core-xy-high",
            ),
        ],
    )
    def test_rejects_common_out_of_range_values(
        self, model: NeuDestInfoModel, axon_max: int, params_update: dict[str, int]
    ):
        with pytest.raises(ValidationError):
            model.model_validate(
                build_v2_dest_info_params(addr_axon=axon_max, **params_update),
                strict=True,
            )

    @pytest.mark.parametrize(("model", "axon_max"), DEST_INFO_CASES)
    def test_rejects_protocol_specific_axon_limit(
        self, model: NeuDestInfoModel, axon_max: int
    ):
        with pytest.raises(ValidationError):
            model.model_validate(
                build_v2_dest_info_params(addr_axon=axon_max + 1), strict=True
            )


class TestNeuHalfAttrsV2:
    @pytest.mark.parametrize(
        ("params_update", "expected_output_type", "expected_vjt"),
        [
            pytest.param({}, OutputType.VALUE, 0, id="defaults"),
            pytest.param(
                {
                    "weight_skew": OfflineNeuRegLimV2.WEIGHT_SKEW_MAX,
                    "weight_address_start": OfflineNeuRegLimV2.WEIGHT_ADDRESS_MAX,
                    "weight_address_end": OfflineNeuRegLimV2.WEIGHT_ADDRESS_MAX,
                    "output_type": OutputType.POTENTIAL,
                    "fold_type": FoldType.FOLDED,
                    "neuron_type": NeuronType.FULL,
                    "vjt": 100,
                },
                OutputType.POTENTIAL,
                100,
                id="boundaries",
            ),
        ],
    )
    def test_offline_accepts_legal_values(
        self,
        params_update: dict[str, Any],
        expected_output_type: OutputType,
        expected_vjt: int,
    ):
        half_attrs = OfflineNeuHalfAttrsV2.model_validate(
            build_v2_half_attrs_params(**params_update), strict=True
        )

        assert half_attrs.output_type == expected_output_type
        assert half_attrs.vjt == expected_vjt

    @pytest.mark.parametrize(
        ("params_update", "expected_output_type", "expected_vjt"),
        [
            pytest.param(
                {"vjt": np.float32(0.0)},
                OnlineOutputType.VALUE,
                cast_fp32_scalar(np.float32(0.0)),
                id="zero-vjt",
            ),
            pytest.param(
                {
                    "weight_skew": OfflineNeuRegLimV2.WEIGHT_SKEW_MAX,
                    "weight_address_start": OfflineNeuRegLimV2.WEIGHT_ADDRESS_MAX,
                    "weight_address_end": OfflineNeuRegLimV2.WEIGHT_ADDRESS_MAX,
                    "output_type": OnlineOutputType.VALUE_AND_POTENTIAL_16BIT,
                    "fold_type": FoldType.FOLDED,
                    "neuron_type": NeuronType.FULL,
                    "vjt": np.float64(1.0 / 3.0),
                },
                OnlineOutputType.VALUE_AND_POTENTIAL_16BIT,
                cast_fp32_scalar(np.float64(1.0 / 3.0)),
                id="boundaries",
            ),
        ],
    )
    def test_online_accepts_legal_values(
        self,
        params_update: dict[str, Any],
        expected_output_type: OnlineOutputType,
        expected_vjt: float,
    ):
        half_attrs = OnlineNeuHalfAttrsV2.model_validate(
            build_online_v2_half_attrs_params(**params_update), strict=True
        )

        assert half_attrs.output_type == expected_output_type
        assert half_attrs.vjt == expected_vjt

    @pytest.mark.parametrize(("model", "builder"), HALF_ATTR_MODEL_CASES)
    @pytest.mark.parametrize(
        "params_update",
        [
            pytest.param(
                {"weight_skew": OfflineNeuRegLimV2.WEIGHT_SKEW_MAX + 1},
                id="weight-skew-high",
            ),
            pytest.param(
                {"weight_address_start": OfflineNeuRegLimV2.WEIGHT_ADDRESS_MAX + 1},
                id="weight-start-high",
            ),
            pytest.param(
                {"weight_address_end": OfflineNeuRegLimV2.WEIGHT_ADDRESS_MAX + 1},
                id="weight-end-high",
            ),
        ],
    )
    def test_rejects_common_out_of_range_values(
        self,
        model: NeuHalfAttrsModel,
        builder: Callable[..., dict[str, Any]],
        params_update: dict[str, int],
    ):
        with pytest.raises(ValidationError):
            model.model_validate(builder(**params_update), strict=True)

    @pytest.mark.parametrize(
        "params_update",
        [
            pytest.param(
                {"output_type": OnlineOutputType.VALUE},
                id="offline-online-output-type",
            ),
            pytest.param({"vjt": np.float32(0.5)}, id="offline-float-vjt"),
        ],
    )
    def test_offline_rejects_protocol_specific_mismatches(
        self, params_update: dict[str, Any]
    ):
        with pytest.raises(ValidationError):
            OfflineNeuHalfAttrsV2.model_validate(
                build_v2_half_attrs_params(**params_update), strict=True
            )

    @pytest.mark.parametrize(
        "params_update",
        [
            pytest.param(
                {"output_type": OutputType.VALUE},
                id="online-offline-output-type",
            ),
            pytest.param(
                {"output_type": 1},
                id="online-raw-output-type",
            ),
        ],
    )
    def test_online_rejects_protocol_specific_mismatches(
        self, params_update: dict[str, Any]
    ):
        with pytest.raises(ValidationError):
            OnlineNeuHalfAttrsV2.model_validate(
                build_online_v2_half_attrs_params(**params_update), strict=True
            )


class TestNeuFullAttrsV2:
    @pytest.mark.parametrize(
        ("part1_update", "part2_update"),
        [
            pytest.param({}, {}, id="defaults"),
            pytest.param(
                {
                    "weight_skew": 1,
                    "weight_address_end": 1,
                    "output_type": OutputType.POTENTIAL,
                    "fold_type": FoldType.FOLDED,
                },
                {
                    "reset_v": OfflineNeuRegLimV2.RESET_V_MAX,
                    "threshold_neg": -100,
                    "threshold_pos": 100,
                    "leak_tau": OfflineNeuRegLimV2.LEAK_TAU_MAX,
                    "leak_v": OfflineNeuRegLimV2.LEAK_V_MAX,
                    "vjt_initial": OfflineNeuRegLimV2.VJT_INITIAL_MAX,
                },
                id="boundaries",
            ),
        ],
    )
    def test_offline_accepts_legal_values(
        self, part1_update: dict[str, Any], part2_update: dict[str, Any]
    ):
        params = build_offline_full_attrs_params(part1_update, part2_update)
        full_attrs = OfflineNeuFullAttrsV2.model_validate(params, strict=True)

        assert full_attrs.neuron_type == NeuronType.FULL
        assert full_attrs.leak_tau == params["leak_tau"]
        assert full_attrs.reset_v == params["reset_v"]
        assert full_attrs.leak_v == params["leak_v"]
        assert full_attrs.vjt_initial == params["vjt_initial"]

    def test_online_accepts_zero_vjt_defaults(self):
        params = build_online_full_attrs_params({"vjt": np.float32(0.0)})
        full_attrs = OnlineNeuFullAttrsV2.model_validate(params, strict=True)

        assert full_attrs.neuron_type == NeuronType.FULL
        assert full_attrs.vjt == cast_fp32_scalar(np.float32(0.0))
        assert full_attrs.leak_tau == params["leak_tau"]

    def test_online_preserves_fp32_carriers(self):
        params = build_online_full_attrs_params(
            {
                "weight_skew": 1,
                "weight_address_end": 1,
                "output_type": OnlineOutputType.VALUE_AND_MAX_POOLING_POSITIONS,
                "fold_type": FoldType.FOLDED,
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
        )
        full_attrs = OnlineNeuFullAttrsV2.model_validate(params, strict=True)

        assert full_attrs.neuron_type == NeuronType.FULL
        assert (
            full_attrs.output_type == OnlineOutputType.VALUE_AND_MAX_POOLING_POSITIONS
        )
        assert full_attrs.vjt == cast_fp32_scalar(params["vjt"])
        assert full_attrs.reset_v == cast_fp32_scalar(params["reset_v"])
        assert full_attrs.threshold_neg == cast_fp32_scalar(params["threshold_neg"])
        assert full_attrs.threshold_pos == cast_fp32_scalar(params["threshold_pos"])
        assert full_attrs.leak_v == cast_fp32_scalar(params["leak_v"])
        assert full_attrs.vjt_initial == cast_fp32_scalar(params["vjt_initial"])

    @pytest.mark.parametrize(
        "params_update",
        [
            pytest.param(
                {"reset_v": OfflineNeuRegLimV2.RESET_V_MIN - 1}, id="reset-v-low"
            ),
            pytest.param(
                {"reset_v": OfflineNeuRegLimV2.RESET_V_MAX + 1}, id="reset-v-high"
            ),
            pytest.param(
                {"leak_tau": OfflineNeuRegLimV2.LEAK_TAU_MIN - 1}, id="leak-tau-low"
            ),
            pytest.param(
                {"leak_tau": OfflineNeuRegLimV2.LEAK_TAU_MAX + 1}, id="leak-tau-high"
            ),
            pytest.param(
                {"leak_v": OfflineNeuRegLimV2.LEAK_V_MIN - 1}, id="leak-v-low"
            ),
            pytest.param(
                {"leak_v": OfflineNeuRegLimV2.LEAK_V_MAX + 1}, id="leak-v-high"
            ),
            pytest.param(
                {"vjt_initial": OfflineNeuRegLimV2.VJT_INITIAL_MIN - 1},
                id="vjt-initial-low",
            ),
            pytest.param(
                {"vjt_initial": OfflineNeuRegLimV2.VJT_INITIAL_MAX + 1},
                id="vjt-initial-high",
            ),
        ],
    )
    def test_offline_rejects_integer_out_of_range_values(self, params_update):
        with pytest.raises(ValidationError):
            OfflineNeuFullAttrsV2.model_validate(
                build_offline_full_attrs_params(part2_update=params_update),
                strict=True,
            )

    @pytest.mark.parametrize(
        "params_update",
        [
            pytest.param(
                {"leak_tau": OnlineNeuRegLimV2.LEAK_TAU_MIN - 1}, id="leak-tau-low"
            ),
            pytest.param(
                {"leak_tau": OnlineNeuRegLimV2.LEAK_TAU_MAX + 1}, id="leak-tau-high"
            ),
        ],
    )
    def test_online_rejects_integer_out_of_range_values(self, params_update):
        with pytest.raises(ValidationError):
            OnlineNeuFullAttrsV2.model_validate(
                build_online_full_attrs_params(
                    part1_update={"vjt": np.float32(0.0)}, part2_update=params_update
                ),
                strict=True,
            )


class TestNeuFoldedAttrsV2:
    @pytest.mark.parametrize("model", FOLDED_PART1_CASES)
    @pytest.mark.parametrize(
        "params_update",
        [
            pytest.param({}, id="default"),
            pytest.param(
                {
                    "fold_range_xy": 2,
                    "fold_range_x": 2,
                    "fold_range_y": 2,
                    "fold_number": 8,
                },
                id="product-8",
            ),
        ],
    )
    def test_part1_accepts_legal_values(
        self, model: NeuFoldedAttrsPart1Model, params_update: dict[str, int]
    ):
        attrs = model.model_validate(
            build_v2_folded_attrs_part1_params(**params_update), strict=True
        )

        assert attrs.fold_number == params_update.get("fold_number", 1)

    @pytest.mark.parametrize("model", FOLDED_PART1_CASES)
    @pytest.mark.parametrize(
        "params_update",
        [
            pytest.param(
                {"fold_range_xy": OfflineNeuRegLimV2.FOLD_RANGE_MAX + 1},
                id="range-high",
            ),
            pytest.param(
                {"fold_skew_xy": OfflineNeuRegLimV2.FOLD_SKEW_MAX + 1},
                id="skew-high",
            ),
            pytest.param(
                {"fold_axon_xy": OfflineNeuRegLimV2.FOLD_AXON_MAX + 1},
                id="axon-high",
            ),
            pytest.param(
                {"fold_number": OfflineNeuRegLimV2.FOLD_NUMBER_MAX + 1},
                id="number-high",
            ),
            pytest.param({"fold_range_xy": 2, "fold_number": 1}, id="product-mismatch"),
        ],
    )
    def test_part1_rejects_invalid_values(
        self, model: NeuFoldedAttrsPart1Model, params_update: dict[str, int]
    ):
        with pytest.raises(ValidationError):
            model.model_validate(
                build_v2_folded_attrs_part1_params(**params_update), strict=True
            )

    @pytest.mark.parametrize(
        ("model", "params_update", "expected"),
        [
            pytest.param(
                OfflineNeuFoldedAttrsV2Part2,
                {"fold_vjt_3": 300, "fold_vjt_2": 200, "fold_vjt_1": 100},
                {
                    "fold_vjt_3": 300,
                    "fold_vjt_2": 200,
                    "fold_vjt_1": 100,
                    "fold_vjt_0": 3,
                },
                id="offline",
            ),
            pytest.param(
                OnlineNeuFoldedAttrsV2Part2,
                {
                    "fold_vjt_3": np.float32(1.0),
                    "fold_vjt_2": np.float64(-2.5),
                    "fold_vjt_1": np.float32(3.25),
                    "fold_vjt_0": np.float64(-4.75),
                },
                {
                    "fold_vjt_3": cast_fp32_scalar(np.float32(1.0)),
                    "fold_vjt_2": cast_fp32_scalar(np.float64(-2.5)),
                    "fold_vjt_1": cast_fp32_scalar(np.float32(3.25)),
                    "fold_vjt_0": cast_fp32_scalar(np.float64(-4.75)),
                },
                id="online",
            ),
        ],
    )
    def test_part2_accepts_protocol_specific_vjt_carriers(
        self,
        model: NeuFoldedAttrsPart2Model,
        params_update: dict[str, Any],
        expected: dict[str, int | float],
    ):
        attrs = model.model_validate(
            build_v2_folded_attrs_part2_params(**params_update), strict=True
        )

        assert attrs.model_dump() == expected


class TestNeuConfV2:
    @pytest.mark.parametrize(
        ("conf_model", "attrs", "dest_info", "expected_attrs", "expected_dest"),
        [
            pytest.param(
                OfflineNeuHalfConfV2,
                build_v2_half_attrs_params(),
                build_v2_dest_info_params(),
                OfflineNeuHalfAttrsV2,
                OfflineNeuDestInfoV2,
                id="offline-half",
            ),
            pytest.param(
                OfflineNeuFullConfV2,
                build_offline_full_attrs_params(),
                build_v2_dest_info_params(),
                OfflineNeuFullAttrsV2,
                OfflineNeuDestInfoV2,
                id="offline-full",
            ),
            pytest.param(
                OnlineNeuHalfConfV2,
                build_online_v2_half_attrs_params(
                    output_type=OnlineOutputType.VALUE_AND_POTENTIAL_16BIT,
                    vjt=0.5,
                ),
                build_v2_dest_info_params(),
                OnlineNeuHalfAttrsV2,
                OnlineNeuDestInfoV2,
                id="online-half",
            ),
            pytest.param(
                OnlineNeuFullConfV2,
                build_online_full_attrs_params(
                    {
                        "output_type": OnlineOutputType.VALUE_AND_MAX_POOLING_POSITIONS,
                        "vjt": 0.5,
                    },
                    {
                        "reset_v": 0.25,
                        "threshold_neg": -1.0,
                        "threshold_pos": 1.0,
                        "leak_v": 0.125,
                        "vjt_initial": 0.0,
                    },
                ),
                build_v2_dest_info_params(),
                OnlineNeuFullAttrsV2,
                OnlineNeuDestInfoV2,
                id="online-full",
            ),
        ],
    )
    def test_conf_wrappers_validate_nested_models(
        self,
        conf_model: type[Any],
        attrs: dict[str, Any],
        dest_info: dict[str, Any],
        expected_attrs: type[Any],
        expected_dest: type[Any],
    ):
        conf = conf_model.model_validate(
            {"attrs": attrs, "dest_info": dest_info}, strict=True
        )

        assert isinstance(conf.attrs, expected_attrs)
        assert isinstance(conf.dest_info, expected_dest)
