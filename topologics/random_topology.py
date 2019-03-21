
from topologics.base_topology import base_topology


class simple_topology(base_topology):

    def reward(self,arr,action):
        if(action==0):
            return 0.1
        if(arr[0]==1):
            return 1
        return 0














