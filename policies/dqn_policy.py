from policies.policy import Policy
import numpy as np


class DqnPolicy(Policy):
    def take_action(self, dqn,state):
        return np.argmax(dqn.get_q(state))
