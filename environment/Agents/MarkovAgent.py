from environment.Agents.Agent import Agent
import random
class MarkovAgent(Agent):
    def __init__(self,in_channel,out_channel,id,policy):
        Agent.__init__(self,in_channel,out_channel,id,policy)# call father constructor
        self.state = "success"
        self.succ_rate = 0.7
        self.fail_rate = 0.3
        self.action_taken = 0
        self.trans=0
    def take_action(self,item):
        trans = random.randint(1, 101)
        if(self.state == "success"):
            if(trans<=self.succ_rate*100):
                action = self.id+1
            else:
                action = 0
        else:
            if(trans<=self.fail_rate*100):
                action=self.id+1
            else:
                action =0
        self.action_taken = action
        self.trans = trans
        return action


    def after_step(self,action,reward):
        if(reward == 1):
            self.state ="success"
        else:
            self.state = "fail"