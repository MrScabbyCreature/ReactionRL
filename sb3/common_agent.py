from ChemRL import ChemRlEnv
from action_wrapper import MoleculeEmbeddingsActionWrapper

from rewards.properties import logP
from rdkit import Chem
import argparse, os

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from stable_baselines3.common.callbacks import BaseCallback
import time
from ChemRL import display_track, TIME_TRACKER
class TimeTrackerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TimeTrackerCallback, self).__init__(verbose)
        self.training_time_start = 0
        self.total_rollout_time = 0
        self.num_rollouts = 0

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.training_time_start = time.time()
        # print("Training start")

    def _on_step(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        self.rollout_start_time = time.time()

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        self.total_rollout_time += (time.time() - self.rollout_start_time)
        self.num_rollouts += 1

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        print()
        print()
        display_track()
        total_training_time = time.time() - self.training_time_start
        print("Total training time =", total_training_time)
        print("Total rollout time =", self.total_rollout_time)
        print("Approx avg training time per rollout = ", (total_training_time - self.total_rollout_time)/self.num_rollouts)

parser = argparse.ArgumentParser()
parser.add_argument("--timesteps", type=int, default=1000000, help="Timesteps to run for")
parser.add_argument("--unique-name", type=str, default="", help="name for saving file")
parser.add_argument("--mode", type=str, choices=["train", "inference"], required=True, help="Train or inference")
parser.add_argument("--model-path-for-inference", type=str, default=None, help="Model path for inference")
parser.add_argument("--reward-metric", type=str, choices=["logp", "qed", "drd2"], default="logp", help="Which metric to optimize for (reward)")

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

        time_tracker_callback = TimeTrackerCallback()

        model.learn(total_timesteps=args.timesteps, callback=[eval_callback, checkpoint_callback, time_tracker_callback])
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
            print(Chem.MolToSmiles(mol), "---", logP(mol))