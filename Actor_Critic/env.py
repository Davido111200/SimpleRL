import gymnasium as gym

env = gym.make('CartPole-v1')

class cartpole(gym.Env):
    def __init__(self) -> None:
        super(env, self).__init__()
        self.env = env
