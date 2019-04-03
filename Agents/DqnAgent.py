
from Agents.Agent import Agent
from buffers.states_buffer import StatesBuffer
import numpy as np

class DqnAgent(Agent):
    def __init__(self,in_channel,out_channel,id,policy,dqn,shape):
        Agent.__init__(self,in_channel,out_channel,id,policy)# call father constructor
        self.dqn = dqn
        self.features = shape[2]
        self.number_of_steps=shape[1]
        self.batch_size = shape[0]
        self.states_buffer = StatesBuffer(self.number_of_steps)

    def take_action(self,item):
        buff = self.states_buffer.buff
        if (len(buff) < self.number_of_steps):
            buff = np.zeros((self.number_of_steps, self.features))
        state_for_input = np.array([buff])
        action = np.argmax(self.dqn.get_q(state_for_input))
        return action


    def after_step(self,state):
        self.states_buffer.add(state)