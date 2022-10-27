import gym
from gym.spaces import Box
import pickle, os
import numpy as np
from utils import MAIN_DIR

class MoleculeEmbeddingsActionWrapper(gym.ActionWrapper):
    def __init__(self, env, ):
        super().__init__(env)
        self.hash_to_embedding_map = pickle.load(open(os.path.join(MAIN_DIR, "datasets/my_uspto/action_embeddings.pickle"), 'rb'))
        embedding_length = self.hash_to_embedding_map[list(self.hash_to_embedding_map.keys())[0]].shape[0]
        self.action_space = Box(low=self.low, high=self.high, shape=(embedding_length,), dtype=np.float32)
        
    
    def action(self, act):
        '''
        Edit what is sent back to the environment.
        '''
        # Get the distances from the actions
        hash_indices = self.applicable_actions.index # self.applicable_actions is part of ChemRLEnv
        embedding_list = np.array([self.hash_to_embedding_map[hash] for hash in hash_indices])

        # Find the closest action
        distance = ((embedding_list-act)**2).sum(axis=1) # sqrt doesnt change argmin
        choice = distance.argmin()

        return_format = ["rsub", "rcen", "rsig", "rsig_cs_indices", "psub", "pcen", "psig", "psig_cs_indices"]
        return self.applicable_actions.loc[hash_indices[choice]][return_format]

if __name__ == "__main__":
    env = gym.make("LunarLanderContinuous-v2")
    wrapped_env = MoleculeEmbeddingsActionWrapper(env, [np.array([1,0]), np.array([-1,0]),
                                        np.array([0,1]), np.array([0,-1])])
    print(wrapped_env.action_space) 