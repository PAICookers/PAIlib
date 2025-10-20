import os
import time
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple, Optional, Union

import pytest

__all__ = ["ParamTestCase", "make_test", "TestCase"]


class ParamTestCase(NamedTuple):
    """Parametrized test cases."""

    argnames: Union[str, tuple[str, ...]]
    argvalues: Sequence[Any]
    ids: Optional[Sequence[str]] = None


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
