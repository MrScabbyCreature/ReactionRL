from stable_baselines3 import A2C
from sb3.common_agent import *
import argparse

# Algo specific args
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
args = parser.parse_args()

# Run training/inference
common_file_name = f"a2c-{args.unique_name}-ts={args.timesteps}-metric={args.reward_metric}"
model_path = f"models/{common_file_name}"

model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=f"./tensorboard/{common_file_name}")
run_training_or_inference(model, model_path, args)