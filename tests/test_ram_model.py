import json
import pytest

from paicorelib import Coord, NeuronDestInfo


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

    dest_info_dict = dest_info.model_dump(by_alias=True)

    with open(ensure_dump_dir / f"ram_model_dest.json", "w") as f:
        json.dump(dest_info_dict, f, indent=4, ensure_ascii=True)
