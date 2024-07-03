import pytest
from pydantic import ValidationError

from paicorelib import Coord, CoordOffset, CoreType, HwConfig


class TestCoord:
    def test_coord_instance(self):
        c = Coord()
        assert c.x == c.y == 0

        with pytest.raises(ValidationError):
            c = Coord(-1, 1)

        with pytest.raises(ValidationError):
            c = Coord(32, 32)

        c = Coord(*(1, 2))
        c = Coord.from_addr(1 << 10 - 1)

        with pytest.raises(ValidationError):
            c = Coord.from_addr(1 << 10)

    @pytest.mark.parametrize(
        "coord, core_type",
        [
            (Coord(0, 0), CoreType.TYPE_OFFLINE),
            (Coord(27, 27), CoreType.TYPE_OFFLINE),
            (Coord(30, 10), CoreType.TYPE_OFFLINE),
            (Coord(29, 30), CoreType.TYPE_ONLINE),
            (Coord(31, 31), CoreType.TYPE_ONLINE),
        ],
    )
    def test_coord_core_type(self, coord, core_type):
        assert coord.core_type is core_type

    def test_op_add(self, monkeypatch):
        c1 = Coord(12, 13)
        c2 = Coord(30, 30)

        # TypeError: Coord + Coord
        with pytest.raises(TypeError):
            s = c1 + c2  # type: ignore

        # Y-priority
        monkeypatch.setattr(HwConfig, "COORD_Y_PRIORITY", True)

        assert Coord(13, 11) == c1 + CoordOffset(1, -2)
        assert Coord(15, 1) == c1 + CoordOffset(2, 20)
        assert Coord(0, 1) == c1 + CoordOffset(-13, 20)
        assert Coord(31, 31) == c1 + CoordOffset(20, -14)

        with pytest.raises(ValueError):
            # sum_x == 32 while y carries (32)
            s = c1 + CoordOffset(20, 19)

        with pytest.raises(ValueError):
            # sum_x == 31 while y carries (32)
            s = c1 + CoordOffset(19, 19)

        with pytest.raises(ValueError):
            # sum_x == -1 while sum_y == 15
            s = c1 + CoordOffset(-13, 2)

        with pytest.raises(ValueError):
            # sum_x == 0 while y borrows (-1)
            s2 = c2 + CoordOffset(-30, -31)

        with pytest.raises(ValueError):
            # sum_x == 32 while y borrows (-2)
            s2 = c2 + CoordOffset(2, -32)

        # X-priority
        monkeypatch.setattr(HwConfig, "COORD_Y_PRIORITY", False)

        assert Coord(0, 31) == c2 + CoordOffset(2, 0)
        assert Coord(31, 31) == c1 + CoordOffset(-13, 19)

        with pytest.raises(ValueError):
            # sum_x == -1 while sum_y == 33
            s = c1 + CoordOffset(-13, 20)

        with pytest.raises(ValueError):
            # sum_x == 32 while sum_y == 31
            s2 = c2 + CoordOffset(1, 12)

    def test_op_iadd(self):
        c1 = Coord(12, 13)

        with pytest.raises(TypeError):
            c1 += [1, 2, 3]  # type: ignore

        with pytest.raises(TypeError):
            c1 += Coord(12, 13)  # type: ignore

        with pytest.raises(TypeError):
            c1 += 1  # type: ignore

        c1 += (-2, 20)
        assert c1 == Coord(11, 1)

        with pytest.raises(ValueError):
            c1 += (1, 2, 3)  # type: ignore

    def test_op_sub(self, monkeypatch):
        c1 = Coord(12, 13)
        c2 = Coord(30, 30)

        assert Coord(13, 11) == c1 - CoordOffset(-1, 2)
        assert CoordOffset(*(18, 17)) == c2 - c1

        # Y-priority
        monkeypatch.setattr(HwConfig, "COORD_Y_PRIORITY", True)

        assert Coord(1, 26) == c1 - CoordOffset(10, 19)  # (2, -6)

        with pytest.raises(ValueError):
            # sub_x == 0 while sub_y == -2
            s2 = c1 - CoordOffset(12, 15)

        # X-priority
        monkeypatch.setattr(HwConfig, "COORD_Y_PRIORITY", False)

        assert Coord(23, 10) == c1 - CoordOffset(21, 2)  # (-9, 11)
        assert Coord(0, 1) == c1 - CoordOffset(-20, 13)  # (32, 0)

        with pytest.raises(ValueError):
            # sub_x == -1 while sub_y == 0
            s2 = c2 - CoordOffset(31, 30)

    def test_op_isub(self, monkeypatch):
        c1 = Coord(12, 13)

        with pytest.raises(TypeError):
            c1 -= [1, 2]  # type: ignore

        with pytest.raises(TypeError):
            c1 -= 1  # type: ignore

        with pytest.raises(TypeError):
            c1 -= Coord(1, 1)

        # X-priority
        monkeypatch.setattr(HwConfig, "COORD_Y_PRIORITY", False)
        c1 -= (21, -2)
        assert isinstance(c1, Coord)
        assert c1 == Coord(23, 14)

        with pytest.raises(ValueError):
            # sub_y = -1
            c1 -= CoordOffset(1, 15)


class TestCoordOffset:
    def test_coordoffset_instance(self):
        c = CoordOffset()
        assert c.delta_x == c.delta_y == 0

        with pytest.raises(ValidationError):
            c = CoordOffset(-32, 1)

        with pytest.raises(ValidationError):
            c = CoordOffset(32, 0)

        c = CoordOffset(*(-1, 31))
        assert c == CoordOffset(-1, 31)

    def test_op_add(self, monkeypatch):
        co1 = CoordOffset(1, 2)
        co2 = co1 + CoordOffset(-31, 1)

        assert isinstance(co2, CoordOffset)
        assert co2 == CoordOffset(-30, 3)

        with pytest.raises(ValidationError):
            co2 = co1 + CoordOffset(31, -2)

        co3 = co1 + Coord(1, 2)
        assert isinstance(co3, Coord)
        assert co3 == Coord(2, 4)

        co4 = Coord(0, 31)
        assert Coord(1, 1) == co4 + CoordOffset.from_offset(2)

        # X-priority
        monkeypatch.setattr(HwConfig, "COORD_Y_PRIORITY", False)

        assert Coord(2, 31) == co4 + CoordOffset.from_offset(2)

    def test_op_iadd(self):
        co1 = CoordOffset(1, 2)
        co1 += CoordOffset(30, 1)

        assert co1 == CoordOffset(31, 3)

        with pytest.raises(ValueError):
            co1 += CoordOffset(30, 1)

        with pytest.raises(ValueError):
            co1 += CoordOffset(-31 * 2 - 1, 0)

    def test_op_sub(self):
        co1 = CoordOffset(1, 1)
        co2 = co1 - CoordOffset(2, 4)

        assert co2 == CoordOffset(-1, -3)

        with pytest.raises(ValueError):
            co3 = CoordOffset(-31, 0)
            co3 = co3 - CoordOffset(1, 1)

    def test_op_isub(self):
        co1 = CoordOffset(10, 10)
        co1 -= CoordOffset(2, 4)

        assert co1 == CoordOffset(8, 6)

        with pytest.raises(TypeError):
            co1 -= 1  # type: ignore

        with pytest.raises(ValidationError):
            co1 -= (-33, 31)

        with pytest.raises(ValidationError):
            co1 -= CoordOffset(-33, 31)

        co1 -= (21, -2)
        assert co1 == CoordOffset(-13, 8)

    def test_from_offset(self):
        co1 = CoordOffset.from_offset(31)
        assert co1 == CoordOffset(0, 31)

        co2 = CoordOffset.from_offset(100)
        assert co2 == CoordOffset(3, 4)

        with pytest.raises(ValidationError):
            co1 = CoordOffset.from_offset(1024)
