import json
from collections import defaultdict
import pickle

from params import EnvConfig, EvalConfig

import pettingzoo
import pettingzoo.classic

import rlcard
import rlcard.agents.pettingzoo_agents

env_config = EnvConfig()
eval_config = EvalConfig()

with open("./outputs/model.pkl", "rb") as f:
    learning_agent = pickle.load(f)

# Create an environment for multi-agent training using pettingzoo
env = pettingzoo.classic.texas_holdem_v4.env(num_players = env_config.num_opponents + 1)
env.seed(0)
env.reset()

agents = {}

# Define the learning agent
device = rlcard.utils.get_device()
learning_agent_name = env.agents[0]
agents[learning_agent_name] = learning_agent

# Define the opponents
for i in range(env_config.num_opponents):
    agents[env.agents[i+1]] = rlcard.agents.pettingzoo_agents.RandomAgentPettingZoo(num_actions=env.action_space(env.agents[i]).n)

# Evaluate
rewards = rlcard.utils.tournament_pettingzoo(env, agents, eval_config.num_games)
learned_agent_reward = rewards[learning_agent_name]

# Save tournament rewards
with open("./outputs/metrics.json", "w") as f:
    json.dump({"tournament_average_reward": learned_agent_reward}, f)


### Simulate and log interesting games

all_games = []

# I don't understand why this is a thing but it is
raw_env = env.env.env.env.env

# Look for interesting games
for i in range(1000):
    game_info = {
        "log" : [],
        "rewards" : {}
    }

    env.reset()
    trajectories = defaultdict(list)

    game_info["log"].append("Hands:")

    for agent_index, agent_name in enumerate(agents):
        game_info["log"].append(f"    {agent_name}: {raw_env.get_state(agent_index)['raw_obs']['hand']}")
    current_public_cards = []
    current_game_rewards = defaultdict(lambda: 0)

    game_info["log"].append("Game Log:")
    for agent_name in env.agent_iter():
        obs, reward, done, _ = env.last()
        trajectories[agent_name].append((obs, reward, done))

        current_game_rewards[agent_name] += reward

        # Check for new cards
        if raw_env.get_state(0)["raw_obs"]["public_cards"] != current_public_cards:
            current_public_cards = raw_env.get_state(0)["raw_obs"]["public_cards"]
            game_info["log"].append(f"    Cards on Table: {current_public_cards}")

        if done:
            action = None
        else:
            action, _ = agents[agent_name].eval_step(obs)
        trajectories[agent_name].append(action)

        env.step(action)

        if action is not None:
            action_names = ['call', 'raise', 'fold', 'check']
            game_info["log"].append(f"    {agent_name}: {action_names[action]}")

    game_info["log"].append(f"Overall player rewards:")
    for agent_name in agents.keys():
        game_info["rewards"][agent_name] = current_game_rewards[agent_name]
        game_info["log"].append(f"    {agent_name}: {current_game_rewards[agent_name]}")

    all_games.append(game_info)

# Save biggest losses
all_games.sort(
    key = lambda game_info: game_info["rewards"][learning_agent_name]
)

with open("outputs/losses_info.txt", "w") as f:
    for game_info in all_games[:eval_config.num_saved_losses]:
        f.write("\n".join(game_info["log"]))
        f.write("\n\n\n")
