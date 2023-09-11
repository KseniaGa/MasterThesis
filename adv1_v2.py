#SIMPE ADVERSARY 

import numpy as np
import pettingzoo
#from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
#from pettingzoo.mpe._mpe_utils.scenario import BaseScenario

#from .._mpe_utils.core import Agent, Landmark, World
#from .._mpe_utils.scenario import BaseScenario

from pettingzoo.utils.conversions import parallel_wrapper_fn

from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from adv1 import Scenario


class raw_env(SimpleEnv):
    def __init__(self, N=8, max_cycles=25, continuous_actions=False):
        scenario = Scenario()
        world = scenario.make_world(N)
        super().__init__(scenario, world, max_cycles, continuous_actions)
        self.metadata['name'] = "adv1_v2"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)
