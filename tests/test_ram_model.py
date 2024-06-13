from contextlib import nullcontext

import numpy as np
import pytest
from pydantic import ValidationError

from paicorelib import *


@pytest.mark.parametrize(
    "params",
    [
        {
            "dest_coords": [Coord(0, 0)],
            "tick_relative": [0] * 100 + [1] * 100,
            "addr_axon": list(range(0, 200)),
            "addr_core_x": 0,
            "addr_core_y": 1,
            "addr_core_x_ex": 0,
            "addr_core_y_ex": 0,
            "addr_chip_x": 0,
            "addr_chip_y": 0,
        },
        {
            "dest_coords": [Coord(0, 0), Coord(1, 0)],
            "tick_relative": [0] * 100,
            "addr_axon": list(range(0, 100)),
            "addr_core_x": 0,
            "addr_core_y": 1,
            "addr_core_x_ex": 0,
            "addr_core_y_ex": 0,
            "addr_chip_x": 0,
            "addr_chip_y": 0,
        },
    ],
)
def test_NeuronDestInfo_instance(ensure_dump_dir, params):
    dest_info = NeuronDestInfo.model_validate(params, strict=True)

    dest_info_dict = dest_info.model_dump_json(indent=2, by_alias=True)

    with open(ensure_dump_dir / "ram_model_dest.json", "w") as f:
        f.write(dest_info_dict)


@pytest.mark.parametrize(
    "params, expectation",
    [
        (
            {
                "reset_mode": RM.MODE_NORMAL,
                "reset_v": -1,
                "leak_comparison": LCM.LEAK_BEFORE_COMP,
                "threshold_mask_bits": 1,
                "neg_thres_mode": NTM.MODE_RESET,
                "neg_threshold": 1,
                "pos_threshold": 0,
                "leak_direction": LDM.MODE_FORWARD,
                "leak_integration_mode": LIM.MODE_DETERMINISTIC,
                "leak_v": np.array([1, 2, 3, 4, 5, 6]),
                "synaptic_integration_mode": SIM.MODE_DETERMINISTIC,
                "bit_truncation": 0,
            },
            nullcontext(),
        ),
        # neg_threshold is a non-negative
        (
            {
                "reset_mode": RM.MODE_NORMAL,
                "reset_v": -1,
                "leak_comparison": LCM.LEAK_BEFORE_COMP,
                "threshold_mask_bits": 1,
                "neg_thres_mode": NTM.MODE_RESET,
                "neg_threshold": -1,
                "pos_threshold": 0,
                "leak_direction": LDM.MODE_FORWARD,
                "leak_integration_mode": LIM.MODE_DETERMINISTIC,
                "leak_v": 1,
                "synaptic_integration_mode": SIM.MODE_DETERMINISTIC,
                "bit_truncation": 1,
            },
            pytest.raises(ValidationError),
        ),
        # threshold_mask_bits is a non-negative
        (
            {
                "reset_mode": RM.MODE_NORMAL,
                "reset_v": 0,
                "leak_comparison": LCM.LEAK_BEFORE_COMP,
                "threshold_mask_bits": -1,
                "neg_thres_mode": NTM.MODE_RESET,
                "neg_threshold": 1 << 10,
                "pos_threshold": 1 << 10,
                "leak_direction": LDM.MODE_REVERSAL,
                "leak_integration_mode": LIM.MODE_STOCHASTIC,
                "leak_v": -1,
                "synaptic_integration_mode": SIM.MODE_STOCHASTIC,
                "bit_truncation": 0,
            },
            pytest.raises(ValidationError),
        ),
        # bit_truncation is a non-negative
        (
            {
                "reset_mode": RM.MODE_NONRESET,
                "reset_v": 1,
                "leak_comparison": LCM.LEAK_AFTER_COMP,
                "threshold_mask_bits": 0,
                "neg_thres_mode": NTM.MODE_SATURATION,
                "neg_threshold": 1 << 10,
                "pos_threshold": 1 << 10,
                "leak_direction": LDM.MODE_FORWARD,
                "leak_integration_mode": LIM.MODE_DETERMINISTIC,
                "leak_v": -1,
                "synaptic_integration_mode": SIM.MODE_STOCHASTIC,
                "bit_truncation": -1,
            },
            pytest.raises(ValidationError),
        ),
    ],
)
def test_NeuronAttrs_instance(ensure_dump_dir, params, expectation):
    with expectation as e:
        attrs = NeuronAttrs.model_validate(params, strict=True)
        attrs_dict = attrs.model_dump_json(indent=2, by_alias=True)

        with open(ensure_dump_dir / "ram_model_attrs.json", "w") as f:
            f.write(attrs_dict)
