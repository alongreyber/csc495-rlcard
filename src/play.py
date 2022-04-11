import json
import pickle
import collections
import shutil

from params import EnvConfig, EvalConfig

import pettingzoo
import pettingzoo.classic

import rlcard
import rlcard.agents.pettingzoo_agents
import rlcard.utils

from utils import LimitholdemRuleAgentPettingZoo, LimitHoldemHumanAgent, augment_observation

def print_header(s: str):
    print(s.center(160, "-"))

env_config = EnvConfig()

# Create an environment for multi-agent training using pettingzoo
env = pettingzoo.classic.texas_holdem_v4.env(num_players = env_config.num_opponents + 1)
env.reset()

agents = {}

print_header("Texas Holdem")
print_header("by Alon Greyber")

print(f"Welcome! This game is designed to be played with {env_config.num_opponents + 1} players")

human_in_game = False

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

    if choice == "h":
        agents[env.agents[i]] = \
            LimitHoldemHumanAgent(
                env.action_space(env.agents[i]).n
            )
        human_in_game = True
    elif choice == "m":
        raise NotImplementedError()
    elif choice == "r":
        agents[env.agents[i]] = \
            rlcard.agents.pettingzoo_agents.RandomAgentPettingZoo(
                num_actions=env.action_space(env.agents[i]).n
            )


raw_env = env.env.env.env.env
env.reset()
trajectories = collections.defaultdict(list)
current_public_cards = []
current_game_rewards = collections.defaultdict(lambda: 0)

print_header("Starting Game")

# Run game
for agent_name in env.agent_iter():
    obs, reward, done, _ = env.last()
    trajectories[agent_name].append((obs, reward, done))

    current_game_rewards[agent_name] += reward
    augment_observation(
        obs, env, agent_name
    )

    # Check for new cards
    if raw_env.get_state(0)["raw_obs"]["public_cards"] != current_public_cards:
        current_public_cards = raw_env.get_state(0)["raw_obs"]["public_cards"]
        rlcard.utils.print_card(current_public_cards)


    if done:
        action = None
    else:
        action, _ = agents[agent_name].eval_step(obs)
    trajectories[agent_name].append(action)

    env.step(action)

    if action is not None:
        action_names = ['call', 'raise', 'fold', 'check']
        print(f"    {agent_name} chose {action_names[action]}")

print_header("Game Over")

# Sort by reward amount
current_game_rewards = {
    k: v for k, v in sorted(current_game_rewards.items(), key=lambda item: item[1], reverse = True)
}

print(f"The winner is {next(iter(current_game_rewards.keys()))}")
print(f"Overall player rewards:")
for agent_name in agents.keys():
    print(f"    {agent_name}: {current_game_rewards[agent_name]}")
