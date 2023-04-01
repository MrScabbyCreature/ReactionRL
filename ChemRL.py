'''
Action format (for now) = ["rsub", "rcen", "rsig", "rsig_cs_indices", "psub", "pcen", "psig", "psig_cs_indices"]
'''

import gym
from gym.spaces import Box
import numpy as np
import pandas as pd
import os

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from action_utils import *
from utils import *
from mol_embedding.chembl_mpnn import mol_to_embedding, atom_to_embedding
from action_wrapper import MoleculeEmbeddingsActionWrapper


start_mols = pd.read_pickle(os.path.join(MAIN_DIR, "datasets/my_uspto/unique_start_mols.pickle"))
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
  metadata = {'render.modes': ['human', 'ansi', "all"], "reward.metrics": ["logp", "qed", "drd2"]}

  def __init__(self, low=-1, high=1, mol_embedding_fn=mol_to_embedding, atom_embedding_fn=atom_to_embedding, reward_metric="logp", goal=False, render_mode="ansi"):
    '''
    low = min value of 
    K = size of molecule embedding
    mol_embedding_fn: function
      Input: Mol
      Output: Some fixed size vector representation of mol
    atom_embedding_fn: function
      Input: Mol, atom_idx
      Output: Some fixed size vector representation of atom
    reward_metric: metric to use as reward # TODO: Allow passing function / hyperparameters for linear combination of multiple properties
      Currently supported: "logp", "qed", "drd2"
    goal(bool): Whether to do goal conditioned RL
    '''
    super(ChemRlEnv, self).__init__()
    self.render_mode = render_mode

    # Define action and observation space 
    self.low = low
    self.high = high
    random_mol = start_mols[0]
    mol_embedding_len = mol_embedding_fn(random_mol).shape[-1]
    atom_embedding_len = atom_embedding_fn(random_mol, 0).shape[-1]
    self.observation_space = Box(low=low, high=high, shape=(mol_embedding_len*(goal+1),), dtype=np.float32)
    self.action_space = Box(low=low, high=high, shape=(4*mol_embedding_len+2*atom_embedding_len,), dtype=np.float32) # TODO: Make default action space some tuple/dict of smile strings and centre(ints). The embedding space will be implemented through wrappers
    self.mol_embedding_fn = mol_embedding_fn
    self.atom_embedding_fn = atom_embedding_fn
    self.reward_metric = reward_metric
    self.goal = goal
    self.replay_buffer = None

  def set_replay_buffer(self, replay_buffer):
    self.replay_buffer = replay_buffer

  def set_reward_metric(self, metric):
    self.reward_metric = metric

  def _get_info(self, mol, target=None):
    return {"mol": mol, "target": target}

  def _state_embedding(self, mol):
    return self.mol_embedding_fn(mol)
  
  def _action_embedding(self, action):
    rsub, rcen, rsig, _, psub, pcen, psig, __ = action
    rsub = Chem.MolFromSmiles(rsub)
    rsig = Chem.MolFromSmiles(rsig)
    psub = Chem.MolFromSmiles(psub)
    psig = Chem.MolFromSmiles(psig)
    embedding = np.concatenate([self.mol_embedding_fn(rsub), self.atom_embedding_fn(rsig, GetAtomWithAtomMapNum(rsig, rcen).GetIdx()), self.mol_embedding_fn(rsig),
                              self.mol_embedding_fn(psub), self.atom_embedding_fn(psig, GetAtomWithAtomMapNum(psig, pcen).GetIdx()), self.mol_embedding_fn(psig),])
    return embedding

  def get_random_action(self):
    return self._action_embedding(self.applicable_actions.sample().iloc[0]) # Can be done with the pickle dict for hash -> embedding (df.index is hash)

  def step(self, action):
    '''
    Execute one time step within the environment
    '''
    try:
      next_state = apply_action(self.state, *action)
    except:
      print("State:", Chem.MolToSmiles(self.state))
      print(action)
      mark_action_invalid(action.name)
      return self.obs, 0, True, {}
    rew = calc_reward(self.state, action, next_state, target=self.target, metric=self.reward_metric)

    # Update trajectory info
    self.trajectory.add_transition(self.state, action, next_state, rew)

    # Update current state
    self.state = next_state
    self.obs = self._state_embedding(self.state)

    # Check if done - Conditions: 
    # (1) Max episode length reached 
    if self.trajectory.get_trajectory_len() >= MAX_EPISODE_LEN:
      done = True
    else:
      done = False

    # (2) No actions applicable on next state 
    # Since we're doing that here, might as well save this info for next timestep
    # For exploration - we can return a random action from this. For exploitation, we can search for nearest neighbor from his list
    self.applicable_actions = get_applicable_actions(self.state)
    if self.applicable_actions.shape[0] == 0:
      done = True

    if self.goal:
      self.obs = np.append(self.obs, self._state_embedding(self.target))

    return self.obs, rew, done, self._get_info(self.state)


  def reset(self, seed=None, return_info=False, options={}):
    global MAX_EPISODE_LEN
    # Reset the state of the environment to an initial state
    super().reset(seed=seed)
    self.trajectory = TrajectoryTracker()

    if "source" in options:
      smiles = options["source"]
      mol = Chem.MolFromSmiles(smiles)
      self.applicable_actions = get_applicable_actions(mol)
    else:
      # Get a random mol to start with - it should have some applicable action (otherwise, there's no point)
      while True: # FIXME: Inefficient to do this in an infinite loop
        smiles = start_mols.sample(random_state=seed).iloc[0]
        mol = Chem.MolFromSmiles(smiles)
        self.applicable_actions = get_applicable_actions(mol)
        if self.applicable_actions.shape[0] == 0:
          if isinstance(seed, int):
            seed+=1
          continue
        else:
          break

    self.state = mol
    
    # Get state embedding to return as the observation
    obs = self._state_embedding(mol)

    # If goal conditioned RL, then construct a target and add to state observation
    self.target = None
    if "target" in options:
      self.target = Chem.MolFromSmiles(options["target"])
    elif self.goal:
      listy_for_replay_buffer = []
      self.target = mol
      MAX_EPISODE_LEN = 1 # np.random.randint(1, 4)
      for _ in range(MAX_EPISODE_LEN):
        # get a random action
        actions = get_applicable_actions(self.target)
        if actions.shape[0] == 0:
          break
        action = actions.sample().iloc[0]

        # apply 
        try:
          listy_for_replay_buffer.append({"state": self.target, "action": action})
          self.target = apply_action(self.target, *action)
          listy_for_replay_buffer[-1]["next_state"] = self.target
        except:
          listy_for_replay_buffer.pop(-1)
          print("State:", Chem.MolToSmiles(self.target))
          print(action)
          mark_action_invalid(action.name)
        
      
      # If replay_buffer, add transition to it, for (hopefully) better training
      if self.replay_buffer is not None:
        # print("POSPOSPOS-1", self.replay_buffer.pos)
        for i, item in enumerate(listy_for_replay_buffer):
          self.replay_buffer.add(np.append(self._state_embedding(item["state"]), self._state_embedding(self.target)), 
                                 np.append(self._state_embedding(item["next_state"]), self._state_embedding(self.target)), 
                                 self._action_embedding(action), 
                                 calc_reward(item["state"], item["action"], item["next_state"], target=self.target, metric=self.reward_metric), 
                                 True if i == len(listy_for_replay_buffer)-1 else False,
                                 [self._get_info(item["state"])])
        # print("POSPOSPOS-2", self.replay_buffer.pos)
          

    if self.target:
      # Concat to self.obs
      obs = np.append(obs, self._state_embedding(self.target))

    # info
    info = self._get_info(mol, self.target)

    return (obs, info) if return_info else obs


  def render(self): 
    '''Render the environment to the terminal or screen(as an image). Preferably do it only at the end of the episode.'''
    # Print on console
    if self.render_mode in ['ansi', "all"]: 
      if self.goal:
        print(f"{bcolors.OKCYAN}Target: {Chem.MolToSmiles(self.target)}{bcolors.ENDC}")
      for s, a, ns, r in self.trajectory.iter_trajectory():
        print(f"{Chem.MolToSmiles(s)} --- {a[2]} || {a[6]} --->{Chem.MolToSmiles(ns)} (r={r})\n")

    # Generate displays
    if self.render_mode in ["human", "all"]:
      reactions = self.trajectory.get_trajectory_as_reactions()
      reaction_images = [Draw.ReactionToImage(rxn) for rxn in reactions]

      cr = 0 # cumulative reward
      for (s, a, ns, r), im in zip(self.trajectory.iter_trajectory(), reaction_images): # imprint reward info
        cr += r
        d1 = ImageDraw.Draw(im)
        if self.goal:
          d1.text((im.width-100, im.height-30), f"sim(ns) = {round(similarity(ns, self.target), 4)}", fill=(0, 0, 0))
        d1.text((im.width-100, im.height-20), f"r = {round(r, 4)}", fill=(0, 0, 0))
        d1.text((im.width-100, im.height-10), f"cr = {round(cr, 4)}", fill=(0, 0, 0))

      im = get_concat_v_multi_resize(reaction_images)
      if self.goal:
        im = get_concat_h_blank(get_concat_v_blank(Image.open("images/goal.png"), Draw.MolToImage(self.target)), im) 
      im.show()


  def close(self):
    pass

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=None, help="Seed to use for env.reset()")
  parser.add_argument("--render", action="store_true", help="Whether to call env.render()")
  parser.add_argument("--render-type", type=str, default="human", choices=ChemRlEnv.metadata["render.modes"], help="Render type. Only works if --render arg is provided.")
  parser.add_argument("--reward-metric", type=str, default="logp", choices=ChemRlEnv.metadata["reward.metrics"], help="The reward metric to use. Cannot provide custom metric using args.")

  # parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
  return parser.parse_args()



if __name__ == "__main__":
  import argparse
  print("###########")
  print("## Start ##")
  print("###########")

  # get args 
  args = get_args()

  # define env
  _env = ChemRlEnv(reward_metric=args.reward_metric, render_mode=args.render_type)
  env = MoleculeEmbeddingsActionWrapper(_env)


  # Check the environment conforms to gym API
  from gym.utils.env_checker import check_env
  # print(check_env(env)) # FIXME: The action input to the env is currently wrong

  # Run a demo
  observation, info = env.reset(seed=args.seed, return_info=True)
  print("Starting mol:", Chem.MolToSmiles(info["mol"]))

  done = False
  while not done:
    # Do not do action_space.sample() because not all actions are applicable on all states
    # Instead use get_random_action(mol) to get a random action applicable on the molecule
    # Otherwise, use action_space.sample() and find a way to convert to discretize it 
    action = env.get_random_action()

    # print("ACTION")
    # print(action)
    observation, reward, done, info = env.step(action)
    # print(observation, reward, done, info)

  if args.render:
    env.render()

  env.close()

