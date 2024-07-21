import json

import pytest
from pydantic import ValidationError

from paicorelib import *


@pytest.mark.parametrize(
    "coord, params",
    [
        (
            Coord(0, 0),
            {
                "name": "Core0",
                "weight_width": WeightWidth.WEIGHT_WIDTH_1BIT,
                "lcn_extension": LCN_EX.LCN_2X,
                "input_width_format": InputWidthFormat.WIDTH_1BIT,
                "spike_width_format": SpikeWidthFormat.WIDTH_1BIT,
                "num_dendrite": 100,
                "max_pooling_en": MaxPoolingEnable.DISABLE,
                "tick_wait_start": 0,
                "tick_wait_end": 0,
                "snn_mode_en": SNNModeEnable.ENABLE,
                "target_lcn": LCN_EX.LCN_1X,
                "test_chip_addr": Coord(0, 0),
            },
        ),
        (
            Coord(0, 1),
            {
                "name": "Core1",
                "weight_width": WeightWidth.WEIGHT_WIDTH_4BIT,
                "lcn_extension": LCN_EX.LCN_2X,
                "input_width_format": InputWidthFormat.WIDTH_1BIT,
                "spike_width_format": SpikeWidthFormat.WIDTH_1BIT,
                "num_dendrite": 500,
                "max_pooling_en": MaxPoolingEnable.DISABLE,
                "tick_wait_start": 0,
                "tick_wait_end": 0,
                "snn_mode_en": SNNModeEnable.ENABLE,
                "target_lcn": LCN_EX.LCN_2X,
                "test_chip_addr": Coord(2, 0),
                "unused_key": 999,
            },
        ),
        (
            Coord(1, 1),
            {
                "name": "Core2",
                "weight_width": WeightWidth.WEIGHT_WIDTH_8BIT,
                "lcn_extension": LCN_EX.LCN_2X,
                "input_width_format": InputWidthFormat.WIDTH_8BIT,
                "spike_width_format": SpikeWidthFormat.WIDTH_8BIT,
                "num_dendrite": 500,
                "max_pooling_en": MaxPoolingEnable.DISABLE,
                "tick_wait_start": 0,
                "tick_wait_end": 0,
                "snn_mode_en": SNNModeEnable.DISABLE,
                "target_lcn": LCN_EX.LCN_4X,
                "test_chip_addr": Coord(2, 1),
            },
        ),
    ],
)
def test_CoreParams_instance(ensure_dump_dir, coord, params):
    params_reg = ParamsReg.model_validate(params, strict=True)
    params_dict = params_reg.model_dump_json(by_alias=True)

    with open(ensure_dump_dir / f"reg_model_{params_reg.name}.json", "w") as f:
        f.write(json.dumps({coord.address: json.loads(params_dict)}, indent=2))


@pytest.mark.parametrize(
    "params",
    [
        {
            # wrong 'tick_wait_end'
            "name": "Core0",
            "weight_width": WeightWidth.WEIGHT_WIDTH_1BIT,
            "lcn_extension": LCN_EX.LCN_2X,
            "input_width_format": InputWidthFormat.WIDTH_1BIT,
            "spike_width_format": SpikeWidthFormat.WIDTH_1BIT,
            "num_dendrite": 100,
            "max_pooling_en": MaxPoolingEnable.DISABLE,
            "tick_wait_start": 1,
            "tick_wait_end": -1,
            "snn_mode_en": SNNModeEnable.ENABLE,
            "target_lcn": LCN_EX.LCN_1X,
            "test_chip_addr": Coord(0, 0),
        },
        {
            # missing key 'test_chip_addr'
            "name": "Core1",
            "weight_width": WeightWidth.WEIGHT_WIDTH_1BIT,
            "lcn_extension": LCN_EX.LCN_2X,
            "input_width_format": InputWidthFormat.WIDTH_1BIT,
            "spike_width_format": SpikeWidthFormat.WIDTH_1BIT,
            "num_dendrite": 500,
            "max_pooling_en": MaxPoolingEnable.DISABLE,
            "tick_wait_start": 1,
            "tick_wait_end": 0,
            "snn_mode_en": SNNModeEnable.ENABLE,
            "target_lcn": LCN_EX.LCN_2X,
            "unused_key": 999,
        },
        {
            # wrong core mode (1,0,1)
            "name": "Core2",
            "weight_width": WeightWidth.WEIGHT_WIDTH_1BIT,
            "lcn_extension": LCN_EX.LCN_1X,
            "input_width_format": InputWidthFormat.WIDTH_8BIT,
            "spike_width_format": SpikeWidthFormat.WIDTH_1BIT,
            "num_dendrite": 500,
            "max_pooling_en": MaxPoolingEnable.DISABLE,
            "tick_wait_start": 1,
            "tick_wait_end": 0,
            "snn_mode_en": SNNModeEnable.ENABLE,
            "target_lcn": LCN_EX.LCN_2X,
            "test_chip_addr": Coord(0, 1),
        },
    ],
)
def test_CoreParams_instance_illegal(params):
    with pytest.raises(ValidationError):
        ParamsReg.model_validate(params, strict=True)
