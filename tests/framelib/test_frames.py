import numpy as np
import pytest
from pydantic import ValidationError

from paicorelib import LCN_EX, Coord
from paicorelib import ReplicationId as RId
from paicorelib import WeightPrecision
from paicorelib.framelib.frame_defs import FrameHeader as FH
from paicorelib.framelib.frame_gen import OfflineFrameGen
from paicorelib.framelib.frames import *
from paicorelib.framelib.utils import ShapeError, TruncationWarning, print_frame


class TestOfflineConfigFrame1:
    @pytest.mark.parametrize(
        "random_seed",
        [
            np.uint64(123456789),
            123456789,
            np.uint8(123),
            np.uint32(1234567),
        ],
    )
    def test_instance(self, random_seed):
        cf = OfflineFrameGen.gen_config_frame1(
            Coord(1, 0), Coord(3, 4), RId(3, 3), random_seed
        )

        assert cf.header == FH.CONFIG_TYPE1
        assert cf.random_seed == random_seed

    def test_instance_userwarning(self):
        with pytest.warns(TruncationWarning):
            cf = OfflineFrameGen.gen_config_frame1(
                Coord(1, 0), Coord(3, 4), RId(3, 3), 1 << 65 - 1
            )

        print()


class TestOfflineConfigFrame2:
    def test_instance(self, gen_random_params_reg_dict):
        params_reg_dict = gen_random_params_reg_dict
        chip_coord, core_coord, rid = Coord(0, 0), Coord(1, 5), RId(2, 2)
        cf = OfflineFrameGen.gen_config_frame2(
            chip_coord, core_coord, rid, params_reg_dict
        )

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
        n_neuron = 3

        cf = OfflineFrameGen.gen_config_frame3(
            chip_coord,
            core_coord,
            rid,
            0,
            n_neuron,
            attr_dict,
            dest_info_dict,
            lcn_ex=LCN_EX.LCN_2X,
            weight_precision=WeightPrecision.WEIGHT_WIDTH_2BIT,
        )

        # print_frame(cf.value)
        assert (
            cf.n_package
            == (1 << LCN_EX.LCN_2X)
            * (1 << WeightPrecision.WEIGHT_WIDTH_2BIT)
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

    def test_instance_illegal_2(
        self, gen_random_neuron_attr_dict, gen_random_dest_info_dict
    ):
        attr_dict = gen_random_neuron_attr_dict
        dest_info_dict = gen_random_dest_info_dict

        # 2. lists are not equal in length
        dest_info_dict["addr_axon"].append(1)
        chip_coord, core_coord, rid = Coord(0, 0), Coord(1, 5), RId(2, 2)

        with pytest.raises(ValueError):
            cf = OfflineFrameGen.gen_config_frame3(
                chip_coord, core_coord, rid, 0, 100, attr_dict, dest_info_dict
            )

    def test_instance_illegal_3(
        self, gen_random_neuron_attr_dict, gen_random_dest_info_dict
    ):
        attr_dict = gen_random_neuron_attr_dict
        dest_info_dict = gen_random_dest_info_dict
        chip_coord, core_coord, rid = Coord(0, 0), Coord(1, 5), RId(2, 2)

        # 3. #N of neurons out of range
        n = 200

        with pytest.raises(ValueError):
            cf = OfflineFrameGen.gen_config_frame3(
                chip_coord, core_coord, rid, 0, n, attr_dict, dest_info_dict
            )


class TestOfflineTestFrame:
    def test_instance(self):
        ti1 = OfflineFrameGen.gen_testin_frame1(Coord(1, 2), Coord(1, 1), RId(0, 0))
        ti2 = OfflineFrameGen.gen_testin_frame2(Coord(1, 2), Coord(1, 1), RId(0, 0))
        ti3 = OfflineFrameGen.gen_testin_frame3(
            Coord(1, 2), Coord(1, 1), RId(0, 0), 1, 1
        )
        ti4 = OfflineFrameGen.gen_testin_frame4(
            Coord(1, 2), Coord(1, 1), RId(0, 0), 1, 1
        )
        to1 = OfflineFrameGen.gen_testout_frame1(
            Coord(1, 2), Coord(1, 1), RId(0, 0), 12345
        )
        to4 = OfflineFrameGen.gen_testout_frame4(
            Coord(1, 2),
            Coord(1, 1),
            RId(0, 0),
            0,
            0,
            np.array([1, 2, 3, 4], dtype=np.uint64),
        )

        v1 = ti1.value
        v2 = ti2.value
        v3 = ti3.value
        v4 = ti4.value
        v5 = to1.value
        v8 = to4.value

        assert v1.ndim > 0
        assert v2.ndim > 0
        assert v3.ndim > 0
        assert v4.ndim > 0
        assert v5.ndim > 0
        assert v8.ndim > 0


class TestOfflineWorkFrame1:
    def test_instance(self):
        wf1 = OfflineWorkFrame1(Coord(1, 2), Coord(3, 4), RId(3, 3), 1, 1, 1)
        v1 = wf1.value

    def test_instance_illegal(self):
        # data out of range
        with pytest.raises(ValueError):
            wf1 = OfflineWorkFrame1(
                Coord(1, 2), Coord(3, 4), RId(3, 3), 0, 0, (1 << 10) - 1
            )

        # incorrect shape
        with pytest.raises(ShapeError):
            wf1 = OfflineWorkFrame1(
                Coord(1, 2),
                Coord(3, 4),
                RId(3, 3),
                0,
                0,
                np.array([1, 2], dtype=np.uint8),
            )

        # axon out of [0, 1151]
        with pytest.raises(ValueError):
            wf1 = OfflineWorkFrame1(Coord(1, 2), Coord(3, 4), RId(3, 3), 1, 1152, 123)


class TestOfflineWorkFrame:
    def test_instance(self):
        n_sync = 10
        wf2 = OfflineFrameGen.gen_work_frame2(Coord(1, 2), n_sync)
        wf3 = OfflineFrameGen.gen_work_frame3(Coord(1, 1))
        wf4 = OfflineFrameGen.gen_work_frame4(Coord(1, 0))

        v2 = wf2.value
        v3 = wf3.value
        v4 = wf4.value

        assert v2.ndim > 0
        assert v3.ndim > 0
        assert v4.ndim > 0


def test_gen_magic_init_frame():
    frames = OfflineFrameGen.gen_magic_init_frame(Coord(1, 2), Coord(3, 4))
    assert frames.size == 3 + 1 + 1
