import gymnasium as gym

env = gym.make('CartPole-v1')

class environment(gym.Env):
    def __init__(self) -> None:
        super(environment, self).__init__()
        self.env = env
