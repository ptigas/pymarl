from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env

from .predator_prey import PredatorPreyCapture
from .particle import Particle
from .matrix_game import NormalFormMatrixGame
from .stag_hunt import StagHunt

import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["pred_prey"] = partial(env_fn, env=PredatorPreyCapture)
REGISTRY["particle"] = partial(env_fn, env=Particle)
REGISTRY["matrix_game"] = partial(env_fn, env=NormalFormMatrixGame)
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)


if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
