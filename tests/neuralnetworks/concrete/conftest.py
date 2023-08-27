import json,pytest

import numpy as np

def pytest_generate_tests(metafunc):
    with open("tests/external_data/computed_data.json") as f:
        tests = json.load(f)
        metafunc.parametrize("t",tests)

@pytest.fixture
def input(t):
    return np.array(t["input"],np.double)

@pytest.fixture
def output(t):
    return  np.array(t["output"],np.double)

