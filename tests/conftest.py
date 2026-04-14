from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    import api.main as m

    monkeypatch.setattr(m, "_load_model", lambda: (MagicMock(), MagicMock(), "cpu"))
    with TestClient(m.app) as c:
        yield c
