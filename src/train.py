import json
import torch
import pickle
from collections import defaultdict

import pettingzoo
import pettingzoo.classic.texas_holdem_v4

import rlcard
import rlcard.agents.pettingzoo_agents

from params import TrainConfig, EnvConfig
from utils import LimitholdemRuleAgentPettingZoo, run_game_pettingzoo

train_config = TrainConfig()
env_config = EnvConfig()

rlcard.utils.set_seed(train_config.seed)

# Create an environment for multi-agent training using pettingzoo
env = pettingzoo.classic.texas_holdem_v4.env(
    num_players = env_config.num_opponents + 1
)
env.seed(train_config.seed)
env.reset()

agents = {}

# Define the learning agent
device = rlcard.utils.get_device()

learning_agent_name = env.agents[0]

learning_agent = rlcard.agents.pettingzoo_agents.DQNAgentPettingZoo(
    num_actions=env.action_space(learning_agent_name).n,
    state_shape=env.observation_space(learning_agent_name)["observation"].shape,
    mlp_layers=[
        train_config.mlp_layer_size
        for _ in range(train_config.mlp_layer_count)
    ],
    device=device,
)

agents[learning_agent_name] = learning_agent

# Define the opponents
for i in range(env_config.num_opponents):
    agents[env.agents[i+1]] = LimitholdemRuleAgentPettingZoo()

reward_info = []

# Train
num_timesteps = 0
for episode in range(train_config.num_training_episodes):
    trajectories = run_game_pettingzoo(env, agents)

    trajectories = rlcard.utils.reorganize_pettingzoo(trajectories)
    num_timesteps += sum([len(t) for t in trajectories.values()])

    for ts in trajectories[learning_agent_name]:
        learning_agent.feed(ts)

    # Compute total reward so we can log it
    if episode % 100 == 0:
        total_reward = \
            sum(ts[2] for ts in trajectories[learning_agent_name])
        reward_info.append(total_reward)

# Save model
with open("./outputs/model.pkl", "wb") as f:
    pickle.dump(learning_agent, f)

# Save plots
with open("./outputs/train_rewards.json", "w") as f:
    json.dump([{"reward" : r} for r in reward_info], f)
