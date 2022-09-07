'''
Action format (for now) = ["rsub", "rcen", "rsig", "rsig_cs_indices", "psub", "pcen", "psig", "psig_cs_indices"]
'''

import gym
from gym.spaces import Box, Discrete, Dict
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from action_utils import *
from utils import *

start_mols = pd.read_pickle("datasets/my_uspto/unique_start_mols.pickle")
MAX_EPISODE_LEN = 5

class TrajectoryTracker:
  def __init__(self) -> None:
    self.trajectory = []

  def add_transition(self, state, action, next_state, reward):
    self.trajectory.append((state, action, next_state, reward))

  def get_trajectory_len(self):
    return len(self.trajectory)

  def iter_trajectory(self):
    for transition in self.trajectory:
      yield transition

  def get_trajectory_as_reactions(self):
    '''
    Converts the trajectory information to sequence of reactions with rsig and psig as reagents
    '''
    reactions = []
    for s, a, ns, r in self.trajectory:
      reactant = Chem.MolToSmiles(s)
      rsig = smiles_without_atom_index(a[2])
      psig = smiles_without_atom_index(a[6])
      product = Chem.MolToSmiles(ns)
      reactions.append(AllChem.ReactionFromSmarts(f'{reactant}>{rsig}.{psig}>{product}',useSmiles=True))
    return reactions

class ChemRlEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human', 'ansi']}

  def __init__(self, low=-1, high=1, K=20, render_mode="ansi"):
    '''
    low = min value of 
    K = size of molecule embedding
    '''
    super(ChemRlEnv, self).__init__()
    self.render_mode = render_mode

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

  def _get_info(self, mol):
    return {"mol": mol}

  def step(self, action):
    '''
    Execute one time step within the environment
    '''
    # FIXME: action should be an embedding and used for reverse lookup. For now it is [rsub, rcen, rsig, rsig_cs_indices, psub, pcen, psig, psig_cs_indices]
    # Note that the cs_indices are not actually part of action (and hence the embedding). They should be purely looked up (they are used for efficiency)
    next_state = apply_action(self.state, *action)
    rew = calc_reward(self.state, action, next_state, metric='logp')

    # Update trajectory info
    self.trajectory.add_transition(self.state, action, next_state, rew)

    # Check if done
    if self.trajectory.get_trajectory_len() >= MAX_EPISODE_LEN:
      done = True
    else:
      done = False

    # Update current state
    self.state = next_state
    self.obs = state_embedding(self.state)

    return self.obs, rew, done, self._get_info(self.state)


  def reset(self, seed=None, return_info=False, options=False):
    # Reset the state of the environment to an initial state
    super().reset(seed=seed)
    self.trajectory = TrajectoryTracker()

    # Get a random mol to start with
    smiles = start_mols.sample(random_state=seed).iloc[0]
    mol = Chem.MolFromSmiles(smiles)
    self.state = mol
    
    # info
    info = self._get_info(mol)

    # Get state embedding to return as the observation
    obs = state_embedding(mol)

    return (obs, info) if return_info else obs


  def render(self): #TODO: update the previous render image 
    '''Render the environment to the terminal or screen(as an image). Preferably do it only at the end of the episode.'''
    # Print on console
    if self.render_mode == 'ansi': 
      for s, a, ns, r in self.trajectory.iter_trajectory():
        print(f"{Chem.MolToSmiles(s)} ---> {Chem.MolToSmiles(ns)} (r={r})")

    # Generate displays
    if self.render_mode == "human":
      reactions = self.trajectory.get_trajectory_as_reactions()
      reaction_images = [Draw.ReactionToImage(rxn) for rxn in reactions]

      im = get_concat_v_multi_resize(reaction_images)
      im.show()

  def close(self):
    pass


if __name__ == "__main__":
  # define env
  env = ChemRlEnv(render_mode="human")

  # Check the environment conforms to gym API
  from gym.utils.env_checker import check_env
  # print(check_env(env))

  # Run a demo
  observation, info = env.reset(seed=42, return_info=True)
  print("Starting mol:", Chem.MolToSmiles(info["mol"]))

  done = False
  while not done:
    action = get_random_action(info["mol"])
    print("ACTION")
    print(action)
    observation, reward, done, info = env.step(action)
    print(observation, reward, done, info)

  env.render()

  env.close()

