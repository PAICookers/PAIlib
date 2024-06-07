import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def ensure_dump_dir():
    p = Path(__file__).parent / "debug"

    if not p.is_dir():
        p.mkdir(parents=True, exist_ok=True)

    yield p


@pytest.fixture(scope="module")
def ensure_dump_dir_and_clean():
    p = Path(__file__).parent / "debug"

    if not p.is_dir():
        p.mkdir(parents=True, exist_ok=True)
    else:
        for f in p.iterdir():
            f.unlink()

    yield p

    # Clean up
    for f in p.iterdir():
        f.unlink()


@pytest.fixture
def cleandir():
    with tempfile.TemporaryDirectory() as newpath:
        old_cwd = os.getcwd()
        os.chdir(newpath)
        yield
        os.chdir(old_cwd)
