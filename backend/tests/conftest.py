import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
_tmp = tempfile.mkdtemp(prefix="spectrarl_test_")
os.environ["SPECTRARL_DATA"] = _tmp
import pytest
from spectrarl import db


@pytest.fixture(scope="session", autouse=True)
def _db():
    db.init(os.path.join(_tmp, "test.db"))
    yield
