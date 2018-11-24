import random
from policies.policy import Policy
class random_policy(Policy):

    def take_action(self,n_channels):
        random_channel = random.randint(0, n_channels)
        action = []
        for i in range(n_channels):
            if(i == random_channel):
                action.append(1)
            else:
                action.append(0)
        return action