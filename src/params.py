
class EnvConfig:
    def __init__(self):
        self.num_opponents = 5


class TrainConfig:
    def __init__(self):
        self.num_training_episodes = 4000
        self.mlp_layer_count = 2
        self.mlp_layer_size = 64
        self.seed = 42

class EvalConfig:
    def __init__(self):
        self.num_games = 1000
        # How many games to save that the learning agent lost
        self.num_saved_losses = 5
        self.seed = 42
