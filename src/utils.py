from collections import defaultdict

from rlcard.utils.pettingzoo_utils import reorganize_pettingzoo
from rlcard.models.limitholdem_rule_models import LimitholdemRuleAgentV1

class LimitholdemRuleAgentPettingZoo(LimitholdemRuleAgentV1):
    all_actions = ['call', 'raise', 'fold', 'check']

    def step(self, state):
        action = super().step(state)
        if action not in state["raw_legal_actions"]:
            # The rule-based agent has some handling of illegal actions, but
            # it's not comprehensive enough so we add to it here
            if "check" in state["raw_legal_actions"]:
                action = "check"
            else:
                # Always legal
                action = "fold"
        # Convert action to index
        return self.all_actions.index(action)

def augment_observation(obs, env, agent_name):
    # Augment observation with raw observation data (not sure why this isn't included)
    obs["raw_obs"] = env.unwrapped.env._extract_state(
        env.unwrapped.env.game.get_state(env.agents.index(agent_name))
    )["raw_obs"]
    obs["raw_legal_actions"] = obs["raw_obs"]["legal_actions"]

def run_game_pettingzoo(env, agents):
    env.reset()
    trajectories = defaultdict(list)
    for agent_name in env.agent_iter():
        obs, reward, done, _ = env.last()
        trajectories[agent_name].append((obs, reward, done))

        augment_observation(
            obs, env, agent_name
        )

        if done:
            action = None
        else:
            action = agents[agent_name].step(obs)
        trajectories[agent_name].append(action)

        env.step(action)

    return trajectories

def tournament_pettingzoo(env, agents, num_episodes):
    total_rewards = defaultdict(float)
    for _ in range(num_episodes):
        trajectories = run_game_pettingzoo(env, agents)
        trajectories = reorganize_pettingzoo(trajectories)
        for agent_name, trajectory in trajectories.items():
            reward = sum([t[2] for t in trajectory])
            total_rewards[agent_name] += reward
    return {k: v / num_episodes for (k, v) in total_rewards.items()}
