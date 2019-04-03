from Agents.Agent import Agent
from policies.random_policy import random_policy
from Channel import Channel
class Simple_Generator(object):
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




