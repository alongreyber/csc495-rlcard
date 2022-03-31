from rlcard.utils.pettingzoo_utils import wrap_state
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
