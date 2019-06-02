
from environment.Agents import Agent
from environment.buffers.states_buffer import StatesBuffer
import numpy as np
import random
class DqnAgent(Agent):
    def __init__(self,in_channel,out_channel,id,policy,dqn,shape):
        Agent.__init__(self,in_channel,out_channel,id,policy)# call father constructor
        self.dqn = dqn
        self.features = shape[2]
        self.number_of_steps=shape[1]
        self.batch_size = shape[0]
        self.states_buffer = StatesBuffer(self.number_of_steps)
        self.counter = 0
    def take_action(self,item):

        if(self.counter < 500*400):
            self.counter +=1
            return random.randint(0, item)
        buff = self.states_buffer.buff
        if (len(buff) < self.number_of_steps):
            buff = np.zeros((self.number_of_steps, self.features))
            print("zero state")
        else:
            iii=9
        state_for_input = np.array([buff])
        prediction = self.dqn.get_q(state_for_input)
        action = np.argmax(prediction)
        return action


    def after_step(self,action,reward):
        state = []
        for i in range(self.features):
            state.append(0)
        state[action] = 1
        state[-1] = reward
        self.states_buffer.add(state)
        print(state,self.id)