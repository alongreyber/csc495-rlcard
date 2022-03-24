
class EnvConfig:
    def __init__(self):
        self.num_opponents = 5


class TrainConfig:
    def __init__(self):
        self.num_training_episodes = 20000
        self.seed = 0

class EvalConfig:
    def __init__(self):
        self.num_games = 200
