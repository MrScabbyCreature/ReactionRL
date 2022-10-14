from ChemRL import ChemRlEnv
from action_wrapper import MoleculeEmbeddingsActionWrapper

from stable_baselines3 import PPO

model_path = "models/ppo"
mode = "train"
# mode = "inference"

if mode == "train":
    env = MoleculeEmbeddingsActionWrapper(ChemRlEnv())
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo/")
    model.learn(total_timesteps=1e5)
    model.save(model_path)

if mode == "inference":
    env = MoleculeEmbeddingsActionWrapper(ChemRlEnv(render_mode="all"))
    model = PPO.load(model_path)

    for i in range(1):
        obs = env.reset()
        done = False
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
        env.render()
        print()
        print()
        print()
        print()