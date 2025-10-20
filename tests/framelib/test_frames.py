import numpy as np
import pytest
from pydantic import ValidationError

from paicorelib import LCN_EX, Coord
from paicorelib import ReplicationId as RId
from paicorelib import WeightWidth as WW
from paicorelib.framelib.frame_defs import FrameHeader as FH
from paicorelib.framelib.frame_gen import OfflineFrameGen, OnlineFrameGen
from paicorelib.framelib.frames import *
from paicorelib.framelib.types import FRAME_DTYPE, LUT_DTYPE
from paicorelib.framelib.utils import ShapeError, TruncationWarning, np2txt
from paicorelib.hw_defs import HwOfflineCoreParams as OffCoreParams
from paicorelib.hw_defs import HwOnlineCoreParams as OnCoreParams
from paicorelib.ram_model import OfflineNeuAttrs as OffNeuAttrs
from paicorelib.ram_model import OfflineNeuDestInfo as OffNeuDestInfo
from paicorelib.ram_model import OnlineNeuAttrs as OnNeuAttrs
from paicorelib.ram_model import OnlineNeuDestInfo as OnNeuDestInfo
from paicorelib.reg_model import OfflineCoreReg, OnlineCoreReg
from paicorelib.routing_defs import _rid_unset

from .gen_testcase import *

RNG = np.random.default_rng()


class TestOfflineFrame:
    @pytest.mark.parametrize(
        "random_seed",
        [np.uint64(123456789), 123456789, np.uint8(123), np.uint32(1234567)],
    )
    def test_cf1(self, random_seed):
        cf = OfflineFrameGen.gen_config_frame1(
            Coord(1, 0), Coord(3, 4), RId(3, 3), random_seed
        )

        assert cf.header == FH.CONFIG_TYPE1

    def test_cf1_userwarning(self):
        with pytest.warns(TruncationWarning):
            cf = OfflineFrameGen.gen_config_frame1(
                Coord(1, 0), Coord(3, 4), RId(3, 3), 1 << 65 - 1
            )

    @pytest.mark.parametrize("core_reg", gen_offline_core_reg_testcase())
    def test_cf2(self, core_reg, ensure_dump_dir):
        chip_coord, core_coord, rid = Coord(0, 0), Coord(1, 5), RId(2, 2)
        cf = OfflineFrameGen.gen_config_frame2(chip_coord, core_coord, rid, core_reg)

        assert cf.header == FH.CONFIG_TYPE2
        assert cf.chip_coord == chip_coord
        assert cf.core_coord == core_coord
        assert cf.rid == rid

        # check value
        np2txt(ensure_dump_dir / "offline_cf2.txt", cf.value)

        core_reg_valid = OfflineCoreReg.model_validate(core_reg)
        cf2 = OfflineFrameGen.gen_config_frame2(
            chip_coord, core_coord, rid, core_reg_valid
        )

        _ = cf2.value  # check value

    @pytest.mark.parametrize("core_reg", gen_offline_core_reg_test_cases())
    def test_cf2_illegal(self, core_reg, monkeypatch):
        chip_coord, core_coord, rid = Coord(0, 0), Coord(1, 5), RId(2, 2)

        # 1. missing keys
        with monkeypatch.context() as m:
            m.delitem(core_reg, "weight_width")

            with pytest.raises(ValidationError, match="weight_width"):
                cf = OfflineConfigFrame2(chip_coord, core_coord, rid, core_reg)

        # 2. type of value is wrong
        with monkeypatch.context() as m:
            m.setitem(core_reg, "snn_en", True)

            with pytest.raises(ValidationError, match="snn_en"):
                cf = OfflineConfigFrame2(chip_coord, core_coord, rid, core_reg)

    @pytest.mark.parametrize("neu_attrs, dest_info", gen_offline_neu_test_cases())
    def test_cf3(self, neu_attrs, dest_info, ensure_dump_dir):
        validate_offline_neu_testcase(neu_attrs, dest_info)
        chip_coord, core_coord, rid = Coord(0, 0), Coord(1, 5), RId(2, 2)
        n_neuron = dest_info["n_neuron"]

        cf = OfflineFrameGen.gen_config_frame3(
            chip_coord, core_coord, rid, 0, n_neuron, neu_attrs, dest_info, 4
        )

        assert (
            cf.n_package
            == (1 << LCN_EX.LCN_2X) * (1 << WW.WEIGHT_WIDTH_2BIT) * 4 * n_neuron
        )

        # check value
        np2txt(ensure_dump_dir / "offline_cf3.txt", cf.value)

        attrs = OffNeuAttrs.model_validate(neu_attrs)
        dest_info = OffNeuDestInfo.model_validate(dest_info)
        cf2 = OfflineFrameGen.gen_config_frame3(
            chip_coord, core_coord, rid, 0, n_neuron, attrs, dest_info, 4
        )

        _ = cf2.value  # check value

    @pytest.mark.parametrize("neu_attrs, dest_info", gen_offline_neu_test_cases())
    def test_cf3_illegal(self, neu_attrs, dest_info, monkeypatch):
        validate_offline_neu_testcase(neu_attrs, dest_info)
        attrs_dict = neu_attrs
        chip_coord, core_coord, rid = Coord(0, 0), Coord(1, 5), RId(2, 2)
        n_neuron = dest_info["n_neuron"]

        # 1. missing keys
        with monkeypatch.context() as m:
            m.delitem(attrs_dict, "reset_mode")

            with pytest.raises(ValidationError, match="reset_mode"):
                cf = OfflineFrameGen.gen_config_frame3(
                    chip_coord, core_coord, rid, 0, n_neuron, attrs_dict, dest_info, 1
                )

        # 2. lists are not equal in length
        with monkeypatch.context() as m:
            temp = dest_info["addr_axon"].copy()
            m.setitem(dest_info, "addr_axon", temp.append(1))

            with pytest.raises(ValueError, match="addr_axon"):
                cf = OfflineFrameGen.gen_config_frame3(
                    chip_coord, core_coord, rid, 0, n_neuron, attrs_dict, dest_info, 1
                )

        # 3. #N of neurons out of range
        n = 200
        with pytest.raises(ValueError, match="tick_relative"):
            cf = OfflineFrameGen.gen_config_frame3(
                chip_coord, core_coord, rid, 0, n, attrs_dict, dest_info, 1
            )

    def test_cf4(self, ensure_dump_dir):
        n_neuron = 100
        wram_weight = RNG.integers(
            0, 256, (n_neuron, OfflineConfigFrame4.N_FRAME_PER_NRAM), dtype=FRAME_DTYPE
        )
        chip_coord, core_coord, rid = Coord(0, 0), Coord(1, 5), _rid_unset()

        cf4 = OfflineFrameGen.gen_config_frame4(
            chip_coord, core_coord, rid, 0, wram_weight.size, wram_weight
        )
        assert (
            cf4.n_package
            == wram_weight.size
            == OfflineConfigFrame4.N_FRAME_PER_NRAM * n_neuron
        )

        # check value
        np2txt(ensure_dump_dir / "offline_cf4.txt", cf4.value)

    def test_test_frames(self):
        ti1 = OfflineFrameGen.gen_testin_frame1(Coord(1, 2), Coord(1, 1), _rid_unset())
        ti2 = OfflineFrameGen.gen_testin_frame2(Coord(1, 2), Coord(1, 1), _rid_unset())
        ti3 = OfflineFrameGen.gen_testin_frame3(
            Coord(1, 2), Coord(1, 1), _rid_unset(), 1, 1
        )
        ti4 = OfflineFrameGen.gen_testin_frame4(
            Coord(1, 2), Coord(1, 1), _rid_unset(), 1, 1
        )
        to1 = OfflineFrameGen.gen_testout_frame1(
            Coord(1, 2), Coord(1, 1), _rid_unset(), 12345
        )
        to4 = OfflineFrameGen.gen_testout_frame4(
            Coord(1, 2),
            Coord(1, 1),
            _rid_unset(),
            4,
            10,
            np.arange(10, dtype=np.uint64),
        )

        v1 = ti1.value
        v2 = ti2.value
        v3 = ti3.value
        v4 = ti4.value
        v5 = to1.value
        v6 = to4.value

        assert v1.ndim > 0
        assert v2.ndim > 0
        assert v3.ndim > 0
        assert v4.ndim > 0
        assert v5.ndim > 0
        assert v6.ndim > 0

    def test_wf1_instance(self):
        wf1 = OfflineWorkFrame1(Coord(1, 2), Coord(3, 4), RId(3, 3), 1, 1, 1)
        v1 = wf1.value

    def test_wf1_instance_illegal(self):
        # data out of range
        with pytest.raises(ValueError, match="data"):
            wf1 = OfflineWorkFrame1(
                Coord(1, 2), Coord(3, 4), RId(3, 3), 0, 0, (1 << 10) - 1
            )

        # incorrect shape
        with pytest.raises(ShapeError, match="data"):
            wf1 = OfflineWorkFrame1(
                Coord(1, 2),
                Coord(3, 4),
                RId(3, 3),
                0,
                0,
                np.array([1, 2], dtype=np.uint8),
            )

        # axon out of range
        max_addr_ax = OffCoreParams.ADDR_AXON_MAX
        with pytest.raises(ValueError, match="axon"):
            wf1 = OfflineWorkFrame1(
                Coord(1, 2), Coord(3, 4), RId(3, 3), 1, max_addr_ax + 1, 123
            )

        # ts out of range
        max_ts = OffCoreParams.N_TIMESLOT_MAX
        with pytest.raises(ValueError, match="timeslot"):
            wf1 = OfflineWorkFrame1(
                Coord(1, 2), Coord(3, 4), RId(3, 3), max_ts + 1, max_addr_ax, 123
            )

    def test_wf1(self, ensure_dump_dir):
        one_input_node = {
            "inp1_1": {
                "addr_core_x": 2,
                "addr_core_y": 2,
                "addr_core_x_ex": 1,
                "addr_core_y_ex": 1,
                "addr_chip_x": 1,
                "addr_chip_y": 0,
                "tick_relative": [0] * 100,
                "addr_axon": list(range(100)),
            }
        }
        data = np.random.randint(0, 256, (100,), dtype=np.uint8)

        wf1 = OfflineFrameGen.gen_work_frame1(one_input_node["inp1_1"], data)
        np2txt(ensure_dump_dir / "offline_wf1.txt", wf1)

    def test_wf1_illegal(self, monkeypatch):
        one_input_node = {
            "inp1_1": {
                "addr_core_x": 2,
                "addr_core_y": 2,
                "addr_core_x_ex": 1,
                "addr_core_y_ex": 1,
                "addr_chip_x": 1,
                "addr_chip_y": 0,
                "tick_relative": [0] * 100,
                "addr_axon": list(range(100)),
            }
        }
        data = np.random.randint(0, 256, (100,), dtype=np.uint8)

        # 1. size mismatch
        with monkeypatch.context() as m:
            m.setitem(one_input_node["inp1_1"], "tick_relative", [0] * 99)

            with pytest.raises(ValueError, match="tick_relative"):
                wf1 = OfflineFrameGen.gen_work_frame1(one_input_node["inp1_1"], data)

        # 2. 'tick_relative' out of range
        max_ts = OffCoreParams.N_TIMESLOT_MAX
        with monkeypatch.context() as m:
            m.setitem(one_input_node["inp1_1"], "tick_relative", [max_ts + 1] * 100)

            with pytest.raises(ValueError, match="tick_relative"):
                wf1 = OfflineFrameGen.gen_work_frame1(one_input_node["inp1_1"], data)

        # 3. 'addr_axon' out of range
        max_addr_ax = OffCoreParams.ADDR_AXON_MAX
        with monkeypatch.context() as m:
            m.setitem(
                one_input_node["inp1_1"],
                "addr_axon",
                list(range(99)) + [max_addr_ax + 1],
            )

            with pytest.raises(ValueError, match="addr_axon"):
                wf1 = OfflineFrameGen.gen_work_frame1(one_input_node["inp1_1"], data)

        # 4. size of data mismatch
        with monkeypatch.context() as m:
            with pytest.raises(ValueError, match="data"):
                wf1 = OfflineFrameGen.gen_work_frame1(
                    one_input_node["inp1_1"], data[:-1]
                )

    def test_wf1_gen_frame_fast(self, ensure_dump_dir):
        frame_dest_info = np.array(
            [
                0b0100_0000100001_00011_11100_00000_00001_000_00000000000_00000000_00000000,
                0b0100_0000100001_00011_11100_00000_00001_000_00000000001_00000000_00000000,
                0b0100_0000100001_00011_11100_00000_00001_000_00000000010_00000000_00000000,
                0b0100_0000100001_00011_11101_00000_00000_000_00000000000_00000000_00000000,
                0b0100_0000100001_00011_11101_00000_00000_000_00000000001_00000000_00000000,
                0b0100_0000100001_00011_11101_00000_00000_000_00000000010_00000000_00000000,
            ],
            dtype=np.uint64,
        )
        data = np.array([1, 2, 0, 0, 5, 6], np.uint8)
        result = OfflineFrameGen.gen_work_frame1_fast(frame_dest_info, data)

        assert result.dtype == np.uint64
        assert result.size == 4  # Only non-zero data will be encoded

        np2txt(ensure_dump_dir / "offline_wf1_gen_fast.txt", result)

    def test_work_frames(self):
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

    def test_gen_magic_init_frame(self, ensure_dump_dir):
        coords = [Coord(0, 0), Coord(3, 4), Coord(7, 8)]

        magic1, magic2 = OfflineFrameGen.gen_magic_init_frame(Coord(1, 2), coords, True)
        assert magic1.size == 2 * 3
        assert magic2.size == 3 * 3

        np2txt(ensure_dump_dir / "magic1.txt", magic1)
        np2txt(ensure_dump_dir / "magic2.txt", magic2)

        magic1, magic2 = OfflineFrameGen.gen_magic_init_frame(
            Coord(1, 2), coords, False
        )
        assert magic1.size == 1 * 3 + 1
        assert magic2.size == 3 * 3


class TestOnlineFrame:
    def test_cf1(self, ensure_dump_dir):
        lut = np.arange(-30, 30, dtype=np.int8)
        cf = OnlineFrameGen.gen_config_frame1(Coord(1, 0), Coord(3, 4), RId(3, 3), lut)

        assert cf.header == FH.CONFIG_TYPE1
        assert len(cf) == OnlineConfigFrame1.N_FRAME_PER_LUT_RAM

        # check value
        np2txt(ensure_dump_dir / "online_cf1.txt", cf.value)

        # Check if the encoded LUT is the same as the original one
        # Frame #6[:2] -> LUT[18][1:0]
        assert (cf.payload[5] >> 28) == lut[18].view(np.uint8) & 0x03

        # Frame #2[11:4] -> LUT[6]
        assert (cf.payload[1] >> 4) & 0xFF == lut[6].view(np.uint8)

        # Frame #10[27:20] -> LUT[34]
        assert (cf.payload[9] >> 20) & 0xFF == lut[34].view(np.uint8)

        # Frame #14[-3:] -> LUT[52][7:4]
        assert (cf.payload[13] & 0x07) == lut[52].view(np.uint8) >> 4

    @pytest.mark.parametrize(
        "lut",
        [
            np.random.randint(-128, 128, size=(50,), dtype=np.int8),
            np.random.randint(0, 256, size=(59,), dtype=np.uint8),
        ],
    )
    def test_cf1_illegal(self, lut):
        with pytest.raises(ValueError):
            _ = OnlineFrameGen.gen_config_frame1(
                Coord(1, 0), Coord(3, 4), RId(3, 3), lut
            )

    @pytest.mark.parametrize("core_reg", gen_online_core_reg_testcase())
    def test_cf2(self, core_reg, ensure_dump_dir):
        chip_coord, core_coord, rid = Coord(0, 0), Coord(30, 28), RId(0, 1)
        cf = OnlineFrameGen.gen_config_frame2(chip_coord, core_coord, rid, core_reg)

        assert cf.header == FH.CONFIG_TYPE2
        assert cf.chip_coord == chip_coord
        assert cf.core_coord == core_coord
        assert cf.rid == rid

        # check value
        np2txt(ensure_dump_dir / "online_cf2.txt", cf.value)

        core_reg_valid = OnlineCoreReg.model_validate(core_reg)
        cf2 = OnlineFrameGen.gen_config_frame2(
            chip_coord, core_coord, rid, core_reg_valid
        )

        _ = cf2.value  # check value

    @pytest.mark.parametrize("core_reg", gen_online_core_reg_testcase())
    def test_cf2_illegal(self, core_reg, monkeypatch):
        chip_coord, core_coord, rid = Coord(0, 0), Coord(30, 28), RId(0, 1)

        # 1. missing keys
        with monkeypatch.context() as m:
            m.delitem(core_reg, "random_seed")
            with pytest.raises(ValidationError, match="random_seed"):
                _ = OnlineFrameGen.gen_config_frame2(
                    chip_coord, core_coord, rid, core_reg
                )

        # 2. invalid values
        with monkeypatch.context() as m:
            m.setitem(core_reg, "inhi_core_x_ex", 10)
            with pytest.raises(ValueError, match="inhi_core_"):
                _ = OnlineFrameGen.gen_config_frame2(
                    chip_coord, core_coord, rid, core_reg
                )

    @pytest.mark.parametrize("neu_attrs, dest_info", gen_online_neu_testcase())
    def test_cf3(self, neu_attrs, dest_info, ensure_dump_dir) -> None:
        validate_online_neu_testcase(neu_attrs, dest_info)
        ww = neu_attrs["weight_width"]
        chip_coord, core_coord, rid = Coord(0, 0), Coord(28, 28), RId(2, 2)
        n_neuron = dest_info["n_neuron"]

        cf = OnlineFrameGen.gen_config_frame3(
            chip_coord, core_coord, rid, 0, n_neuron, neu_attrs, dest_info, ww
        )

        # check value
        np2txt(ensure_dump_dir / "online_cf3.txt", cf.value)

        attrs = OnNeuAttrs.model_validate(neu_attrs, context={"weight_width": ww})
        dest_info = OnNeuDestInfo.model_validate(dest_info)
        cf2 = OnlineFrameGen.gen_config_frame3(
            chip_coord, core_coord, rid, 0, n_neuron, attrs, dest_info, ww
        )

        _ = cf2.value  # check value

    @pytest.mark.parametrize("neu_attrs, dest_info", gen_online_neu_testcase())
    def test_cf3_illegal(self, neu_attrs, dest_info, monkeypatch):
        validate_online_neu_testcase(neu_attrs, dest_info)
        ww = neu_attrs["weight_width"]
        chip_coord, core_coord, rid = Coord(0, 0), Coord(28, 28), RId(2, 2)
        n_neuron = dest_info["n_neuron"]

        # 1. missing keys
        with monkeypatch.context() as m:
            m.delitem(neu_attrs, "pos_threshold")

            with pytest.raises(ValidationError, match="pos_threshold"):
                _ = OnlineFrameGen.gen_config_frame3(
                    chip_coord, core_coord, rid, 0, n_neuron, neu_attrs, dest_info, ww
                )

        # 2. lists are not equal in length
        with monkeypatch.context() as m:
            temp = dest_info["addr_axon"].copy()
            m.setitem(dest_info, "addr_axon", temp.append(1))

            with pytest.raises(ValueError, match="addr_axon"):
                _ = OnlineFrameGen.gen_config_frame3(
                    chip_coord, core_coord, rid, 0, n_neuron, neu_attrs, dest_info, ww
                )

        # 3. #N of neurons out of range
        n = 200
        with pytest.raises(ValueError, match="tick_relative"):
            _ = OnlineFrameGen.gen_config_frame3(
                chip_coord, core_coord, rid, 0, n, neu_attrs, dest_info, ww
            )

    def test_cf4(self, ensure_dump_dir):
        ww = 8
        n_neuron = 64

        wram_weight = RNG.integers(
            0,
            256,
            (n_neuron, OnlineConfigFrame4.N_FRAME_PER_NRAM),
            dtype=FRAME_DTYPE,
        )
        chip_coord, core_coord, rid = Coord(0, 0), Coord(31, 30), _rid_unset()

        cf4 = OnlineFrameGen.gen_config_frame4(
            chip_coord, core_coord, rid, 0, wram_weight.size, wram_weight
        )
        assert (
            cf4.n_package
            == wram_weight.size
            == OnlineConfigFrame4.N_FRAME_PER_NRAM * n_neuron
        )

        # check value
        np2txt(ensure_dump_dir / "online_cf4.txt", cf4.value)

    def test_test_frames(self):
        ti1 = OnlineFrameGen.gen_testin_frame1(Coord(1, 2), Coord(31, 31), _rid_unset())
        ti2 = OnlineFrameGen.gen_testin_frame2(Coord(1, 2), Coord(31, 31), _rid_unset())
        ti3 = OnlineFrameGen.gen_testin_frame3(
            Coord(1, 2), Coord(31, 31), _rid_unset(), 1, 1
        )
        ti4 = OnlineFrameGen.gen_testin_frame4(
            Coord(1, 2), Coord(31, 31), _rid_unset(), 1, 1
        )
        to1 = OnlineFrameGen.gen_testout_frame1(
            Coord(1, 2),
            Coord(30, 30),
            _rid_unset(),
            np.ones((OnCoreParams.LUT_LEN,), dtype=LUT_DTYPE),
        )
        to4 = OnlineFrameGen.gen_testout_frame4(
            Coord(1, 2),
            Coord(29, 31),
            _rid_unset(),
            0,
            100,
            np.arange(100, dtype=np.uint64),
        )

        v1 = ti1.value
        v2 = ti2.value
        v3 = ti3.value
        v4 = ti4.value
        v5 = to1.value
        v6 = to4.value

        assert v1.ndim > 0
        assert v2.ndim > 0
        assert v3.ndim > 0
        assert v4.ndim > 0
        assert v5.ndim > 0
        assert v6.ndim > 0

    def test_wf1_1_instance(self):
        wf1 = OnlineWorkFrame1_1(Coord(1, 2), Coord(31, 29), RId(3, 3), 1, 1)
        v1 = wf1.value

    def test_wf1_illegal(self):
        # axon out of [0, 1151]
        max_addr_ax = OnCoreParams.ADDR_AXON_MAX
        with pytest.raises(ValueError, match="axon"):
            wf1 = OnlineWorkFrame1_1(
                Coord(1, 2), Coord(31, 29), RId(3, 0), 1, max_addr_ax + 1
            )

        max_ts = OnCoreParams.N_TIMESLOT_MAX
        with pytest.raises(ValueError, match="timeslot"):
            wf1 = OnlineWorkFrame1_1(
                Coord(1, 2), Coord(31, 29), RId(3, 0), max_ts + 1, max_addr_ax
            )

    def test_wf1_1(self, ensure_dump_dir):
        one_input_node = {
            "inp1_1": {
                "addr_core_x": 28,
                "addr_core_y": 29,
                "addr_core_x_ex": 1,
                "addr_core_y_ex": 0,
                "addr_chip_x": 0,
                "addr_chip_y": 0,
                "tick_relative": [0, 0, 0, 1, 1, 1],
                "addr_axon": [0, 1, 2, 3, 4, 5],
            }
        }

        wf1 = OnlineFrameGen.gen_work_frame1_1(one_input_node["inp1_1"])
        np2txt(ensure_dump_dir / "online_wf1_1.txt", wf1)

    def test_wf1_1_illegal(self, monkeypatch):
        one_input_node = {
            "inp1_1": {
                "addr_core_x": 28,
                "addr_core_y": 29,
                "addr_core_x_ex": 1,
                "addr_core_y_ex": 0,
                "addr_chip_x": 0,
                "addr_chip_y": 0,
                "tick_relative": [0] * 100 + [1] * 100 + [2] * 100,
                "addr_axon": list(range(300)),
            }
        }

        # 1. size mismatch
        with monkeypatch.context() as m:
            temp = one_input_node["inp1_1"]["addr_axon"].copy()
            m.setitem(one_input_node["inp1_1"], "addr_axon", temp.append(1))

            with pytest.raises(ValueError, match="addr_axon"):
                wf1 = OnlineFrameGen.gen_work_frame1_1(one_input_node["inp1_1"])

        # 2. 'tick_relative' out of range
        with monkeypatch.context() as m:
            m.setitem(one_input_node["inp1_1"], "tick_relative", list(range(300)))

            with pytest.raises(ValueError, match="tick_relative"):
                wf1 = OnlineFrameGen.gen_work_frame1_1(one_input_node["inp1_1"])

        # 3. 'addr_axon' out of range
        max_addr_ax = OnCoreParams.ADDR_AXON_MAX
        with monkeypatch.context() as m:
            m.setitem(
                one_input_node["inp1_1"],
                "addr_axon",
                list(range(299)) + [max_addr_ax + 1],
            )

            with pytest.raises(ValueError, match="addr_axon"):
                wf1 = OnlineFrameGen.gen_work_frame1_1(one_input_node["inp1_1"])

    def test_work_frames(self):
        wf1_2 = OnlineFrameGen.gen_work_frame1_2(
            Coord(1, 0), Coord(31, 31), _rid_unset()
        )
        wf1_3 = OnlineFrameGen.gen_work_frame1_3(
            Coord(1, 0), Coord(31, 31), _rid_unset()
        )
        wf1_4 = OnlineFrameGen.gen_work_frame1_4(
            Coord(1, 0), Coord(31, 31), _rid_unset()
        )

        n_sync = 10
        wf2 = OnlineFrameGen.gen_work_frame2(Coord(31, 29), n_sync)
        wf3 = OnlineFrameGen.gen_work_frame3(Coord(31, 30))
        wf4 = OnlineFrameGen.gen_work_frame4(Coord(28, 30))

        v1 = wf1_2.value
        v2 = wf1_3.value
        v3 = wf1_4.value
        v4 = wf2.value
        v5 = wf3.value
        v6 = wf4.value

        assert v1.ndim > 0
        assert v2.ndim > 0
        assert v3.ndim > 0
        assert v4.ndim > 0
        assert v5.ndim > 0
        assert v6.ndim > 0
