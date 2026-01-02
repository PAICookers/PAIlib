import random
import string
from collections.abc import Iterable
from typing import TypeVar

import numpy as np


def _mask(mask_bit: int) -> int:
    return (1 << mask_bit) - 1


IT = TypeVar("IT", bound=Iterable)


def range_check(param: IT, field: str, min_val: int, max_val: int) -> IT:
    if any(item > max_val or item < min_val for item in param):
        raise ValueError(f"parameter '{field}' out of range [{min_val}, {max_val}].")

    return param


def range_check_unsigned(param: IT, field: str, max_val: int) -> IT:
    return range_check(param, field, 0, max_val)


def gen_random_string(length: int = 8) -> str:
    chars = string.ascii_letters + string.digits
    return "".join(random.choice(chars) for _ in range(length))


def ndarray_serializer(value: int | np.ndarray) -> int | list[int]:
    return value if isinstance(value, int) else value.tolist()
