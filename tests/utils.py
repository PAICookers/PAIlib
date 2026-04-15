import os
import time
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pytest
from numpy.typing import DTypeLike

from paicorelib.coordinate import CoordZXYOffset
from paicorelib.core_defs import LCN_EX
from paicorelib.core_defs_v2 import (
    AddPotentialMode,
    CSCAccelerateMode,
    DataSign,
    DataWidth,
    PoolingMode,
    SNNMode,
    ZeroOutputMode,
)
from paicorelib.neuron_defs import ResetMode
from paicorelib.neuron_defs_v2 import (
    FoldType,
    LateralInhibitionMode,
    LeakAddMode,
    LeakMultiComparisonOrder,
    LeakMultiInputMode,
    LeakMultiMode,
    NeuronType,
    OutputType,
    ThresholdNegMode,
    ThresholdPosMode,
    WeightCompressType,
)
from paicorelib.routing_hexa import AERPacketZXYCopy
from paicorelib.utils import _mask

__all__ = ["ParamTestCase", "make_test", "TestCase"]


class ParamTestCase(NamedTuple):
    """Parametrized test cases."""

    argnames: str | tuple[str, ...]
    argvalues: Sequence[Any]
    ids: Sequence[str] | None = None


def make_test(
    cases: ParamTestCase,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable) -> Callable:
        return pytest.mark.parametrize(cases.argnames, cases.argvalues, ids=cases.ids)(
            func
        )

    return decorator


class TestCase:
    """Base class for test data."""

    __test__ = False


@contextmanager
def measure_time(desc: str) -> Generator[None, Any, None]:
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"{desc} executed in: {elapsed:.2f} secs")


def file_not_exist_fail(_fp: str | Path) -> None:
    """Raise a `pytest.fail` if the file does not exist."""
    fp = Path(_fp)
    if Path.is_file(fp) and not fp.exists():
        pytest.fail(f"test file {fp} does not exist.")


def gen_random_array(
    shape: tuple[int, ...],
    dtype: DTypeLike,
    rng: np.random.Generator | None = None,
    sparse_ratio: float = 0.0,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    if isinstance(dtype, bool):
        arr = rng.integers(0, 1, shape, dtype, endpoint=True)
    else:
        arr = rng.integers(
            np.iinfo(dtype).min, np.iinfo(dtype).max, shape, dtype, endpoint=True
        )

    if sparse_ratio == 0.0:
        return arr

    num_zeros = int(arr.size * sparse_ratio)
    if num_zeros > 0:
        flat_indices = rng.choice(arr.size, size=num_zeros, replace=False)
        arr.flat[flat_indices] = 0

    return arr


CI_INDICATORS = ["CI", "CI_ENV", "GITHUB_ACTIONS"]


def is_ci_env() -> bool:
    return any(os.getenv(var) for var in CI_INDICATORS)


def make_dump_dir(
    test_path: Path, temp_path_fac: pytest.TempPathFactory, dir_name: str = "debug"
) -> Path:
    if is_ci_env():
        p = temp_path_fac.mktemp(dir_name)
    else:
        p = test_path / dir_name

    if not p.is_dir():
        p.mkdir(parents=True, exist_ok=True)
    else:
        for f in p.iterdir():
            f.unlink(missing_ok=True)

    return p


def with_overrides(base: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    params = base.copy()
    params.update(overrides)
    return params


def build_v2_core_reg_params(**overrides: Any) -> dict[str, Any]:
    base = {
        "snn_ann": SNNMode.SNN,
        "max_pooling": PoolingMode.AVERAGE,
        "add_potential": AddPotentialMode.NORMAL,
        "zero_output": ZeroOutputMode.DISABLE,
        "input_sign": DataSign.UNSIGNED,
        "input_width": DataWidth.WIDTH_1BIT,
        "output_sign": DataSign.UNSIGNED,
        "output_width": DataWidth.WIDTH_1BIT,
        "weight_sign": DataSign.UNSIGNED,
        "weight_width": DataWidth.WIDTH_1BIT,
        "lcn": LCN_EX.LCN_1X,
        "target_lcn": LCN_EX.LCN_1X,
        "axon_skew": 0,
        "neuron_number": 100,
        "test_core_xy": 0,
        "test_core_x": 0,
        "test_core_y": 0,
        "global_send": 0,
        "csc_accelerate": CSCAccelerateMode.DISABLE,
        "global_receive": 0,
        "thread_number": 1,
        "busy_cycle": 10,
        "delay_cycle": 2,
        "width_cycle": 2,
        "tick_start": 0,
        "tick_duration": 100,
        "tick_initial": 0,
    }
    return with_overrides(base, **overrides)


def build_v2_dest_info_params(**overrides: Any) -> dict[str, Any]:
    base = {
        "tick_relative": 1,
        "addr_axon": 1,
        "addr_core_xy": 0,
        "addr_core_x": 0,
        "addr_core_y": 0,
        "addr_copy_xy": 0,
        "addr_copy_x": 0,
        "addr_copy_y": 0,
    }
    return with_overrides(base, **overrides)


def build_v2_half_attrs_params(**overrides: Any) -> dict[str, Any]:
    base = {
        "weight_skew": 0,
        "weight_address_start": 0,
        "weight_address_end": 0,
        "output_type": OutputType.VALUE,
        "fold_type": FoldType.UNFOLDED,
        "neuron_type": NeuronType.HALF,
        "vjt": 0,
    }
    return with_overrides(base, **overrides)


def build_v2_full_attrs_part2_params(**overrides: Any) -> dict[str, Any]:
    base = {
        "reset_mode": ResetMode.MODE_NORMAL,
        "reset_v": 0,
        "threshold_neg_mode": ThresholdNegMode.FIRE,
        "threshold_pos_mode": ThresholdPosMode.FIRE,
        "threshold_neg": 0,
        "threshold_pos": 0,
        "lateral_inhibition": LateralInhibitionMode.DISABLE,
        "leak_multi_sequence": LeakMultiComparisonOrder.BEFORE_COMPARE,
        "leak_multi_input": LeakMultiInputMode.DISABLE,
        "leak_multi_mode": LeakMultiMode.DISABLE,
        "leak_add_mode": LeakAddMode.FORWARD,
        "leak_tau": 0,
        "leak_v": 0,
        "weight_compress": WeightCompressType.DENSE,
        "vjt_initial": 0,
    }
    return with_overrides(base, **overrides)


def build_v2_folded_attrs_part1_params(**overrides: Any) -> dict[str, Any]:
    base = {
        "fold_range_xy": 1,
        "fold_range_x": 1,
        "fold_range_y": 1,
        "fold_skew_xy": 0,
        "fold_skew_x": 0,
        "fold_skew_y": 0,
        "fold_axon_xy": 0,
        "fold_axon_x": 0,
        "fold_axon_y": 0,
        "fold_number": 1,
    }
    return with_overrides(base, **overrides)


def build_v2_folded_attrs_part2_params(**overrides: Any) -> dict[str, Any]:
    base = {
        "fold_vjt_3": 0,
        "fold_vjt_2": 1,
        "fold_vjt_1": 2,
        "fold_vjt_0": 3,
    }
    return with_overrides(base, **overrides)


def build_v2_packet_route() -> tuple[CoordZXYOffset, AERPacketZXYCopy]:
    return CoordZXYOffset(1, 1, 1), AERPacketZXYCopy(0, 1, -1)


def build_v2_weight_array(
    size: int,
    weight_width: int,
    signed: bool,
    rng: np.random.Generator,
    sparse_ratio: float = 0.0,
) -> np.ndarray:
    if signed:
        dtype = np.int8
        max_value = _mask(weight_width - 1)
        min_value = -(max_value + 1)
    else:
        dtype = np.uint8
        min_value, max_value = 0, _mask(weight_width)

    weight = rng.integers(min_value, max_value, size=size, dtype=dtype)

    if sparse_ratio > 0 and weight.size > 0:
        n_zero = int(weight.size * sparse_ratio)
        if n_zero > 0:
            indices = rng.choice(weight.size, size=n_zero, replace=False)
            weight[indices] = 0

    return weight


def bit_field(value: int | np.ndarray | np.generic, offset: int, mask: int) -> int:
    scalar = np.asarray(value).item()
    return (int(scalar) >> offset) & mask
