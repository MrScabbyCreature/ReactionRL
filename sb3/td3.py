from stable_baselines3 import TD3
from sb3.common_agent import *
import argparse

# Run training/inference
common_file_name = f"td3-{args.unique_name}-ts={args.timesteps}-metric={args.reward_metric}"
model_path = f"models/{common_file_name}"

model = TD3("MlpPolicy", env, verbose=1, tensorboard_log=f"./tensorboard/{common_file_name}")
run_training_or_inference(model, model_path, args)