from abc import ABC, abstractmethod
"""
a user who wishes to use the environment should inherit from tpology and implement a topology with reward function for his usecase. 
the topology should be provided by the users generator
"""
class base_topology(ABC):
    agents = []
    channels = []
    reward_function = None


    def __init__(self, agents, channels, allow_bidirection = True):
        self.allow_bidirection = allow_bidirection
        self.agents = agents
        self.channels = channels


    def calculate_acknowledge(self,channels_current_step_state,number_of_agents):
        """in the base topology each agent can transmit only on one channel in each action if a user want
        to use a different network protocol he should implement a new topology that extends base_topology and overide this function"""
        ack_arr = []  # to tell which users got acknowledge for sending
        for i in range(number_of_agents):
            ack_arr.append(0)
        for item in channels_current_step_state:
            if (len(item) == 1):
                ack_arr[item[0]] = 1
        return ack_arr

    def check_valid_action(self, action):
        """in the base topology each action should transmit only on one channel"""
        # found_trams_channel = False
        # for channel in action:
        #     if (channel == 1):
        #         if (found_trams_channel == True):
        #             return False
        #         found_trams_channel = True
        # return True
        return action >=0 and action <=len(self.channels)

    @abstractmethod
    def reward(self,arr):
        pass




