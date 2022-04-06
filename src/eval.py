import json
from collections import defaultdict
import random
import pickle

from params import EnvConfig, EvalConfig
from utils import LimitholdemRuleAgentPettingZoo

import pettingzoo
import pettingzoo.classic.texas_holdem_v4

import rlcard
import rlcard.agents.pettingzoo_agents

env_config = EnvConfig()
eval_config = EvalConfig()

rlcard.utils.set_seed(eval_config.seed)

with open("./outputs/model.pkl", "rb") as f:
    learning_agent = pickle.load(f)

# Create an environment for multi-agent training using pettingzoo
env = pettingzoo.classic.texas_holdem_v4.env(num_players = env_config.num_opponents + 1)
env.seed(eval_config.seed)
env.reset()

agents = {}

# Define the learning agent
device = rlcard.utils.get_device()
learning_agent_name = env.agents[0]
agents[learning_agent_name] = learning_agent

opponent_type_rewards = {}

for opponent_type in eval_config.opponent_types:
    # Define the opponents
    for i in range(env_config.num_opponents):
        if opponent_type == "random":
            agents[env.agents[i+1]] = rlcard.agents.pettingzoo_agents.RandomAgentPettingZoo(num_actions=env.action_space(env.agents[i]).n)
        if opponent_type == "rule":
            agents[env.agents[i+1]] = LimitholdemRuleAgentPettingZoo()
        if opponent_type == "mixed":
            # Half rule, half random
            if i % 2 == 0:
                agents[env.agents[i+1]] = rlcard.agents.pettingzoo_agents.RandomAgentPettingZoo(num_actions=env.action_space(env.agents[i]).n)
            else:
                agents[env.agents[i+1]] = LimitholdemRuleAgentPettingZoo()

    # Evaluate
    rewards = rlcard.utils.tournament_pettingzoo(env, agents, eval_config.num_games)
    learned_agent_reward = rewards[learning_agent_name]
    opponent_type_rewards[opponent_type + "_average_reward"] = learned_agent_reward

# Save tournament rewards
with open("./outputs/metrics.json", "w") as f:
    json.dump(opponent_type_rewards, f)


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
