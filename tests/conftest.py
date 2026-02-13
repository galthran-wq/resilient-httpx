import pytest


@pytest.fixture
def proxy_list():
    return [
        "http://proxy1:8080",
        "http://proxy2:8080",
        "http://proxy3:8080",
    ]
