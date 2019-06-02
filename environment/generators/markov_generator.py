from environment.generators.simple_generator import Simple_Generator
from environment.Agents.MarkovAgent import MarkovAgent
from environment.policies.random_policy import random_policy

class Markov_Generator(Simple_Generator):
    def generate_agents(self):
        agents_list = []
        for i in range(self.n_agents):
            agent_arr = []
            for j in range(self.n_channels):
                agent_arr.append(1)
            agents_list.append(MarkovAgent(agent_arr, agent_arr, i, random_policy()))
        return agents_list