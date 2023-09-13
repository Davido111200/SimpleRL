import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env("Walker2d-v4", n_envs=4)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=1000000, log_interval=1, progress_bar=True)
model.save("walker")

del model # remove to demonstrate saving and loading

model = PPO.load("walker")

obs = vec_env.reset()
n_ts =0
while True:
    n_ts += 1
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    print(rewards)
    if n_ts == 100:
        break

    # vec_env.render("human")