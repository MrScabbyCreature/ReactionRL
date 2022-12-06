from ChemRL import ChemRlEnv
from action_wrapper import MoleculeEmbeddingsActionWrapper

from rewards.properties import *
from rdkit import Chem
import argparse, os

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

parser = argparse.ArgumentParser()
parser.add_argument("--timesteps", type=int, default=1000000, help="Timesteps to run for")
parser.add_argument("--unique-name", type=str, default="", help="name for saving file")
parser.add_argument("--mode", type=str, choices=["train", "inference"], required=True, help="Train or inference")
parser.add_argument("--model-path-for-inference", type=str, default=None, help="Model path for inference")
parser.add_argument("--reward-metric", type=str, choices=["logp", "qed", "drd2", "SA"], default="logp", help="Which metric to optimize for (reward)")

env = MoleculeEmbeddingsActionWrapper(ChemRlEnv(render_mode="all"))

def run_training_or_inference(model, path, args):
    env.set_reward_metric(args.reward_metric)

    if args.mode == "train":
        eval_callback = EvalCallback(env, log_path=path+"/", eval_freq=1000, n_eval_episodes=100,
                             deterministic=True, render=False)

        checkpoint_callback = CheckpointCallback(
                            save_freq=args.timesteps//5,
                            save_path=path + "/",
                            name_prefix="checkpoint",
                            )
        model.learn(total_timesteps=args.timesteps, callback=[eval_callback, checkpoint_callback])
        model.save(path+"/model")

    elif args.mode == "inference":
        mol_list = []
        if args.model_path_for_inference:
            path = args.model_path_for_inference
        model = model.__class__.load(os.path.join(path, "model"))

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
            print(Chem.MolToSmiles(mol), f"\n--- {round(logP(mol), 4)}(logp), {round(qed(mol), 4)}(qed), {round(drd2(mol), 4)}(drd2), {round(SA(mol), 4)}(SA)\n")