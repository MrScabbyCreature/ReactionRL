from ChemRL import ChemRlEnv
from action_wrapper import MoleculeEmbeddingsActionWrapper

from stable_baselines3 import SAC
from rewards.properties import logP
from rdkit import Chem
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--timesteps", type=int, default=1e6, help="Timesteps to run for")
parser.add_argument("--unique-name", type=str, default="", help="name for saving file")
parser.add_argument("--mode", type=str, required=True, help="Train or inference")
parser.add_argument("--model-path-for-inference", type=str, default=None, help="Model path for inference")
args = parser.parse_args()


common_file_name = f"sac-{args.unique_name}-lr={args.lr}-ts={args.timesteps}"
model_path = f"models/{common_file_name}"
mode = args.mode

if mode == "train":
    env = MoleculeEmbeddingsActionWrapper(ChemRlEnv())
    model = SAC("MlpPolicy", env, verbose=1, learning_rate=args.lr, tensorboard_log=f"./tensorboard/{common_file_name}")
    model.learn(total_timesteps=args.timesteps)
    model.save(model_path)


mol_list = []
if mode == "inference":
    env = MoleculeEmbeddingsActionWrapper(ChemRlEnv(render_mode="all"))
    if args.model_path_for_inference:
        model_path = args.model_path_for_inference
    model = SAC.load(model_path)

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