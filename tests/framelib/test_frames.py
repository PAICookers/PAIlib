import numpy as np
import pytest
from pydantic import ValidationError

from paicorelib import Coord, ReplicationId as RId, LCN_EX, WeightPrecision
from paicorelib.framelib.frame_gen import OfflineFrameGen
from paicorelib.framelib.frames import *
from paicorelib.framelib.frame_defs import FrameHeader as FH
from paicorelib.ram_model import NeuronAttrs


class TestOfflineConfigFrame1:
    @pytest.mark.parametrize(
        "random_seed",
        [
            np.uint64(123456789),
            123456789,
            np.array([123456789], np.uint64),
            np.uint8(123),
            np.uint32(1234567),
        ],
    )
    def test_instance(self, random_seed):
        cf = OfflineConfigFrame1(Coord(1, 0), Coord(3, 4), RId(3, 3), random_seed)

        assert cf.header == FH.CONFIG_TYPE1
        assert cf.random_seed == random_seed

        # TODO decode here

    @pytest.mark.parametrize(
        "random_seed",
        [np.array([12345, 6789], np.uint64)],
    )
    def test_instance_illegal(self, random_seed):
        with pytest.raises(TypeError):
            # Must call `int()`
            cf = OfflineConfigFrame1(Coord(1, 0), Coord(3, 4), RId(3, 3), random_seed)


class TestOfflineConfigFrame2:
    def test_instance(self, gen_random_params_reg_dict):
        params_reg_dict = gen_random_params_reg_dict
        chip_coord, core_coord, rid = Coord(0, 0), Coord(1, 5), RId(2, 2)
        cf = OfflineConfigFrame2(chip_coord, core_coord, rid, params_reg_dict)

        assert cf.header == FH.CONFIG_TYPE2
        assert cf.chip_coord == chip_coord
        assert cf.core_coord == core_coord
        assert cf.rid == rid

    def test_instance_illegal(self, gen_random_params_reg_dict):
        params_reg_dict = gen_random_params_reg_dict

        # 1. missing keys
        params_reg_dict.pop("weight_width")

        chip_coord, core_coord, rid = Coord(0, 0), Coord(1, 5), RId(2, 2)

        with pytest.raises(ValidationError):
            cf = OfflineConfigFrame2(chip_coord, core_coord, rid, params_reg_dict)

        # 2. type of value is wrong
        params_reg_dict = gen_random_params_reg_dict
        params_reg_dict["snn_en"] = True
        with pytest.raises(ValidationError):
            cf = OfflineConfigFrame2(chip_coord, core_coord, rid, params_reg_dict)


class TestOfflineConfigFrame3:
    def test_instance(self, gen_random_neuron_attr_dict, gen_random_dest_info_dict):
        attr_dict = gen_random_neuron_attr_dict
        dest_info_dict = gen_random_dest_info_dict
        chip_coord, core_coord, rid = Coord(0, 0), Coord(1, 5), RId(2, 2)
        n_neuron = 100

        cf = OfflineFrameGen.gen_config_frame3(
            chip_coord,
            core_coord,
            rid,
            0,
            n_neuron,
            attr_dict,
            dest_info_dict,
            lcn_ex=LCN_EX.LCN_2X,
            weight_precision=WeightPrecision.WEIGHT_WIDTH_8BIT,
        )

        assert (
            cf.n_package
            == (1 << LCN_EX.LCN_2X)
            * (1 << WeightPrecision.WEIGHT_WIDTH_8BIT)
            * 4
            * n_neuron
        )

    def test_instance_illegal_1(
        self, gen_random_neuron_attr_dict, gen_random_dest_info_dict
    ):
        attr_dict = gen_random_neuron_attr_dict
        dest_info_dict = gen_random_dest_info_dict
        chip_coord, core_coord, rid = Coord(0, 0), Coord(1, 5), RId(2, 2)

        # 1. missing keys
        attr_dict.pop("reset_mode")

        with pytest.raises(ValidationError):
            cf = OfflineFrameGen.gen_config_frame3(
                chip_coord, core_coord, rid, 0, 100, attr_dict, dest_info_dict
            )

    def test_instance_illegal_2(self, gen_random_neuron_attr_dict, gen_random_dest_info_dict):
        attr_dict = gen_random_neuron_attr_dict
        dest_info_dict = gen_random_dest_info_dict
        
        # 2. lists are not equal in length
        dest_info_dict["addr_axon"].append(1)
        chip_coord, core_coord, rid = Coord(0, 0), Coord(1, 5), RId(2, 2)

        with pytest.raises(ValueError):
            cf = OfflineFrameGen.gen_config_frame3(
                chip_coord, core_coord, rid, 0, 100, attr_dict, dest_info_dict
            )

    def test_instance_illegal_3(self, gen_random_neuron_attr_dict, gen_random_dest_info_dict):
        attr_dict = gen_random_neuron_attr_dict
        dest_info_dict = gen_random_dest_info_dict
        chip_coord, core_coord, rid = Coord(0, 0), Coord(1, 5), RId(2, 2)
        
        # 3. #N of neurons out of range
        n = 200

        with pytest.raises(ValueError):
            cf = OfflineFrameGen.gen_config_frame3(
                chip_coord, core_coord, rid, 0, n, attr_dict, dest_info_dict
            )


def test_gen_magic_init_frame():
    frames = OfflineFrameGen.gen_magic_init_frame(Coord(1, 2), Coord(3, 4))

    print()
