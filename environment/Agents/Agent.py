"""
user of the environment should implement an agent that fits his usecase and inherit from this base agent
"""
class Agent(object):

    def __init__(self,in_channel,out_channel,id,policy):
        self.in_channel = in_channel
        self.out_channel = out_channel
       # self.take_action = take_action
        self.id = id
        self.policy = policy

    def take_action(self,item):
        print (self.id)
        return self.id+1#self.policy.take_action(item)

    def get_in_channel(self):
        return self.in_channel

    def get_out_channel(self):
        return self.out_channel

    def after_step(self,state,reward):
        pass


