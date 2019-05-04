from generators.simple_generator import Simple_Generator
from policies.dqn_policy import DqnPolicy
from Agents.DqnAgent import DqnAgent

class Dqn_Generator(Simple_Generator):
    dqn=None
    shape = None

    def add_items(self,item1,item2):
        self.dqn = item1
        self.shape = item2

    def generate_agents(self):
        #dp = DqnPolicy()
        agents_list = []
        for i in range(self.n_agents):
            agent_arr = []
            for j in range(self.n_channels):
                agent_arr.append(1)
            agents_list.append(DqnAgent(in_channel=list(agent_arr),
                                        out_channel=list(agent_arr),
                                        id=i,
                                        policy= DqnPolicy(),
                                        dqn=self.dqn,
                                        shape=self.shape))
        return agents_list