from environment.Agents import Agent
from environment.policies import random_policy
from environment.Channel import Channel
class Simple_Generator(object):
    """
    each generator provides the users
    channels and topologics for the environment
    """
    def __init__(self,n_agents,n_channels):
        self.n_agents = n_agents
        self.n_channels = n_channels

    def generate_agents(self):
        rp = random_policy()
        agents_list = []
        for i in range(self.n_agents):
            agent_arr = []
            for j in range(self.n_channels):
                agent_arr.append(1)
            agents_list.append(Agent(agent_arr,agent_arr,i,random_policy()))
        return agents_list

    def generate_channels(self):
        channels_list = []
        for i in range(self.n_channels):
            channels_list.append(Channel(1))
        return channels_list


    def add_items(self,item1,item2):
        pass




