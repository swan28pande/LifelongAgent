import copy
from functools import lru_cache

import pytest
import yaml

from generator_v2 import config, simulator
from generator_v2.schema import PersonaSpec, load_ladder, load_spec


@lru_cache
def ladder():
    return load_ladder(config.LADDER_PATH)


@lru_cache
def raw(user: str) -> dict:
    return yaml.safe_load((config.PERSONA_DIR / f"{user}.yaml").read_text())


def make_spec(user: str, mutate=None) -> PersonaSpec:
    data = copy.deepcopy(raw(user))
    if mutate:
        mutate(data)
    spec = PersonaSpec.model_validate(data)
    spec.difficulty = ladder().users[spec.user_id]
    return spec


@pytest.fixture(scope="session")
def u1():
    return load_spec(config.PERSONA_DIR / "u1.yaml", ladder())


@pytest.fixture(scope="session")
def u5():
    return load_spec(config.PERSONA_DIR / "u5.yaml", ladder())


@pytest.fixture(scope="session")
def w1(u1):
    return simulator.simulate(u1)


@pytest.fixture(scope="session")
def w5(u5):
    return simulator.simulate(u5)
