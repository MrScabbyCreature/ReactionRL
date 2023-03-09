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
parser.add_argument("--reward-metric", type=str, choices=["logp", "qed", "drd2", "SA", "sim"], default="logp", help="Which metric to optimize for (reward)")
parser.add_argument("--goal-conditioned", action="store_true", help="goal")
parser.add_argument("--source-mol", type=str, default=None, help="Source molecule")
parser.add_argument("--target-mol", type=str, default=None, help="Target molecule")
args = parser.parse_args()

# Check for cannot give "sim" (similarity) as a reward without GCRL
if args.reward_metric == "sim":
    assert args.goal_conditioned, "Cannot give a similarity reward without a target (GCRL)"

env = MoleculeEmbeddingsActionWrapper(ChemRlEnv(reward_metric=args.reward_metric, goal=args.goal_conditioned, render_mode="all"))

def run_training_or_inference(model, path, args):
    if args.mode == "train":
        eval_callback = EvalCallback(env, log_path=path+"/", eval_freq=1000, n_eval_episodes=100,
                             deterministic=True, render=False)

        checkpoint_callback = CheckpointCallback(
                            save_freq=args.timesteps//5,
                            save_path=path + "/",
                            name_prefix="checkpoint",
                            )
        env.set_replay_buffer(model.replay_buffer)
        model.learn(total_timesteps=args.timesteps, callback=[eval_callback, checkpoint_callback])
        model.save(path+"/model")

    elif args.mode == "inference":
        mol_list = []
        if args.model_path_for_inference:
            path = args.model_path_for_inference
        model = model.__class__.load(os.path.join(path, "model"))

        # Send source and target mols if provided
        options = {}
        if args.source_mol:
            options.update({"source": args.source_mol})
            options.update({"target": args.target_mol})
        for i in range(1):
            obs, info = env.reset(return_info=True, options=options)
            mol_list.append(info["mol"])
            done = False
            while not done:
                action, _state = model.predict(obs) #, deterministic=True) # <-- Using this returns the action with highest probability
                obs, reward, done, info = env.step(action)
                mol_list.append(info["mol"])
            env.render()
            print()
            print()
            print()
            print()

        for mol in mol_list:
            print(Chem.MolToSmiles(mol), f"\n--- {round(logP(mol), 4)}(logp), {round(qed(mol), 4)}(qed), {round(drd2(mol), 4)}(drd2), {round(SA(mol), 4)}(SA)" + \
                f", {round(similarity(mol, env.target), 4)}(sim)" if args.goal_conditioned else ""  + "\n")