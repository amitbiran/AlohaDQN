from environment.environment import envorinment
from topologics.base_topology import base_topology
import random

class simple_topology(base_topology):

    def reward(self,arr,action):
        if(arr[action]==1):
            return 1
        return 0














