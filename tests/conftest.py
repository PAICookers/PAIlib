import os
import pytest
import tempfile

from .utils import make_dump_dir


@pytest.fixture(scope="module")
def ensure_dump_dir(request, tmp_path_factory):
    p = make_dump_dir(request.path.parent, tmp_path_factory)
    yield p


@pytest.fixture(scope="module")
def ensure_dump_dir_and_clean(request, tmp_path_factory):
    p = make_dump_dir(request.path.parent, tmp_path_factory)
    yield p
    for f in p.iterdir():
        f.unlink(missing_ok=True)


@pytest.fixture
def cleandir():
    with tempfile.TemporaryDirectory() as newpath:
        old_cwd = os.getcwd()
        os.chdir(newpath)
        yield
        os.chdir(old_cwd)
