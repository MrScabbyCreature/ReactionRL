from ChemRL import ChemRlEnv
from action_wrapper import MoleculeEmbeddingsActionWrapper

from stable_baselines3 import A2C
from rewards.properties import logP
from rdkit import Chem

model_path = "models/a2c"
# mode = "train"
mode = "inference"

if mode == "train":
    env = MoleculeEmbeddingsActionWrapper(ChemRlEnv())
    model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./a2c/")
    model.learn(total_timesteps=1e5)
    model.save(model_path)

mol_list = []
if mode == "inference":
    env = MoleculeEmbeddingsActionWrapper(ChemRlEnv(render_mode="all"))
    model = A2C.load(model_path)

    for i in range(1):
        obs, info = env.reset(return_info=True)
        mol_list.append(info["mol"])
        done = False
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            mol_list.append(info["mol"])
        env.render()
        print()
        print()
        print()
        print()


for mol in mol_list:
    print(Chem.MolToSmiles(mol), "---", logP(mol))