import random
from policies.policy import Policy
class random_policy(Policy):

    def take_action(self,n_channels):
        random_channel = random.randint(0, n_channels)
        return random_channel