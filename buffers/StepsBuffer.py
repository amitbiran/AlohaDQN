import numpy as np
from buffers.states_buffer import StatesBuffer

class StepsBuffer():
    def __init__(self,number_of_steps, buffer_size=50000):
            self.actions_trace = StatesBuffer(number_of_steps)
            self.rewards_trace = StatesBuffer(number_of_steps)
            for ii in range(number_of_steps):
                self.rewards_trace.add(0)
                self.actions_trace.add(0)
            self.buffer = []
            self.buffer_size = buffer_size


    def add(self, experience):
        exp=experience[0]
        self.actions_trace.add(exp[1])
        self.rewards_trace.add(exp[2])
        arr = []
        for item in exp:
            arr.append(item)
        arr.append(list(self.actions_trace.buff))
        arr.append(list(self.rewards_trace.buff))
        arr = np.reshape(np.array(arr),(1,7))
        if len(self.buffer) + len(arr) >= self.buffer_size:
            self.buffer[0:(len(arr) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(arr)
        a=0