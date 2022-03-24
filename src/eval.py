import json
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

# env.set_agents(agents)

# Evaluate
rewards = rlcard.utils.tournament_pettingzoo(env, agents, eval_config.num_games)
learned_agent_reward = rewards[learning_agent_name]

with open("./outputs/metrics.json", "w") as f:
    json.dump({"tournament_average_reward": learned_agent_reward}, f)
