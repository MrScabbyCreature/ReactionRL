from tracemalloc import start
import gym
from gym.spaces import Box, Discrete, Dict
import numpy as np
import pandas as pd
from rdkit import Chem
from action_utils import get_random_action

start_mols = pd.read_pickle("datasets/my_uspto/unique_start_mols.pickle")

class ChemRlEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, low=-1, high=1, K=20):
    '''
    low = min value of 
    K = size of molecule embedding
    '''
    super(ChemRlEnv, self).__init__()
    # Define action and observation space # TODO: IMPLEMENT YOUR OWN ACTION SPACE CLASS (for obs space too?)
    self.observation_space = Box(low=low, high=high, shape=(K,), dtype=np.float32)
    self.action_space = Dict({
                                "rsig": Box(low=low, high=high, shape=(K,), dtype=np.float32),
                                "rsub": Box(low=low, high=high, shape=(K,), dtype=np.float32),
                                "rcen": Discrete(100),
                                "psig": Box(low=low, high=high, shape=(K,), dtype=np.float32),
                                "psub": Box(low=low, high=high, shape=(K,), dtype=np.float32),
                                "pcen": Discrete(100),
                            }) 

    self.obs, self.state = self.reset(return_info=True)

  def _get_obs(self):
    # Convert self.mol to embedding and return
    return None # TODO
  
  def _get_info(self, mol=None):
    return {"mol": mol}

  def step(self, action):
    # Execute one time step within the environment
    self.timestep += 1

    

  def reset(self, seed=None, return_info=False, options=False):
    # Reset the state of the environment to an initial state
    super().reset(seed=seed)
    self.timestep = 1

    smiles = start_mols.sample(random_state=seed).iloc[0]
    mol = Chem.MolFromSmiles(smiles)
    info = self._get_info()

    # TODO: Convert mol to an embedding
    embedding = self.observation_space.sample() # THIS IS WRONG - FOR TESTING ONLY

    return (embedding, info) if return_info else mol


  def render(self, mode='human', close=False):
    # Render the environment to the screen
    ...

if __name__ == "__main__":
  # define env
  env = ChemRlEnv()

  # Check the environment conforms to gym API
  from gym.utils.env_checker import check_env
  # print(check_env(env))

  # Run a demo
  observation, info = env.reset(seed=42, return_info=True)

  for _ in range(10):
      action = get_random_action(info["mol"])
      observation, reward, done, info = env.step(action)

      if done:
          observation, info = env.reset(return_info=True)

  env.close()

