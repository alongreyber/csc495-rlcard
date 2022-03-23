import torch
import pickle

from utils import import_submodules

import pettingzoo
import pettingzoo.classic

import rlcard
import rlcard.agents.pettingzoo_agents

from params import TrainConfig, EnvConfig

train_config = TrainConfig()
env_config = EnvConfig()

# Create an environment for multi-agent training using pettingzoo
env = pettingzoo.classic.texas_holdem_v4.env(num_players = env_config.num_opponents + 1)
env.seed(train_config.seed)
env.reset()

agents = {}

# Define the learning agent
device = rlcard.utils.get_device()

learning_agent_name = env.agents[0]
learning_agent = rlcard.agents.pettingzoo_agents.DQNAgentPettingZoo(
    num_actions=env.action_space(learning_agent_name).n,
    state_shape=env.observation_space(learning_agent_name)["observation"].shape,
    mlp_layers=[64,64],
    device=device,
)
agents[learning_agent_name] = learning_agent

# Define the opponents
for i in range(env_config.num_opponents):
    agents[env.agents[i+1]] = rlcard.agents.pettingzoo_agents.RandomAgentPettingZoo(num_actions=env.action_space(env.agents[i]).n)

# Train
num_timesteps = 0
for episode in range(train_config.num_training_episodes):
    trajectories = rlcard.utils.run_game_pettingzoo(env, agents, is_training=True)
    trajectories = rlcard.utils.reorganize_pettingzoo(trajectories)
    num_timesteps += sum([len(t) for t in trajectories.values()])

    for ts in trajectories[learning_agent_name]:
        learning_agent.feed(ts)

# Save model
with open("./models/model.pkl", "wb") as f:
    pickle.dump(learning_agent, f)
