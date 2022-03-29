import json
import pickle

from params import EnvConfig, EvalConfig

import pettingzoo
import pettingzoo.classic

import rlcard
import rlcard.agents.pettingzoo_agents

env_config = EnvConfig()

# Create an environment for multi-agent training using pettingzoo
env = pettingzoo.classic.texas_holdem_v4.env(num_players = env_config.num_opponents + 1)
env.reset()

agents = {}

print("--------------- Texas Holdem ---------------")
print("-------------- by Alon Greyber -------------")

print(f"Welcome! This game is designed to be played with {env_config.num_opponents + 1} players")

for i in range(env_config.num_opponents + 1):
    while True:
        print(f"Please enter a choice for player {i}")
        print("[h] - Human")
        print("[m] - Model")
        print("[r] - Random")
        choice = input("[h/m/r]: ")
        if choice in ["h", "m", "r"]:
            break
        print(f"Invalid choice {choice}")

    # TODO finish this
