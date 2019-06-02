
from environment.topologics.base_topology import base_topology


class simple_topology(base_topology):
    punishment = -1
    last_0_reward = 0
    last_0_action = 0
    last_try = 0
    last_try_count = 1
    def reward(self,arr,action,id):
        if(id==0):

            if(action==0 and arr[1]==1 and arr[2]==1):
                self.last_0_reward = 0.5
                self.last_0_action = action
                self.punishment = -1
                return 0.5
            if(action == 0 and (arr[1] == 0 or arr[2] == 0)):
                self.last_0_reward = -0.5
                self.last_0_action = action
                self.punishment = -1
                return -0.5
            if(arr[0]==0):
                #to encourage agent not to stick to only one channel we will give him panelty if he tries the same channel again and again
                if(action == self.last_try):
                    self.last_try_count += 1
                else:
                    self.last_try_count = 1
                if(self.last_0_reward == -1 and self.last_0_action == action):
                    self.punishment *= 4
                    self.last_0_action = action
                    self.last_try = action
                    self.last_0_reward = -1
                    return self.punishment
                else:
                    self.punishment = -1
                    self.last_0_reward =-1
                    self.last_try = action
                    self.last_0_action = action
                    return -1
            if(arr[0]==1):
                #to encourage agent not to stick to only one channel we will give him panelty if he tries the same channel again and again
                self.punishment=-1
                self.last_0_action = action
                self.last_0_reward = 2
                if (action == self.last_try):
                    self.last_try_count += 1
                    return 2#0.2
                else:
                    self.last_try_count = 1
                    return 2
            return 0
        else:
            if(arr[id]==1):
                return 1
            else:
                return 0















