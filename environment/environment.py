from generators.simple_generator import Simple_Generator
from topologics.random_topology import simple_topology
from Qnetwork import Qnetwork
class environment(object):

    #constructor
    def __init__(self, verbose=False,num_of_users=3,num_of_channels=2,dqn = None):
        self.verbose = verbose
        generator = Simple_Generator(n_agents=num_of_users-1,n_channels=num_of_channels,env=self)
        self.agents = generator.generate_agents()
        self.channels = generator.generate_channels()
        self.number_of_channels = len(self.channels)
        topology = simple_topology(self.agents,self.channels)
        self.reward = topology.reward
        self.calculate_acknowledge = topology.calculate_acknowledge
        self.check_valid_action = topology.check_valid_action
        self.actions = self.number_of_channels+1

    def step(self, action):
        actions = [action]
        for agent in self.agents:
            actions.append(agent.take_action(len(self.channels)))

        channel_state =[]#array of arrays, in each array is a list of all users tring to transmit on that channel.
        channel_state.append([])
        for cahnnel in self.channels:
            channel_state.append([])

        # iterate through the channels
        for channel_number in range(self.actions):
            #iterate through the actions per channel
            for j in range(len(actions)):
                if(channel_number==actions[j]):#if someone tried to transmit on the channel add his id to the channel state array
                    channel_state[channel_number].append(j)


        ack_arr = self.calculate_acknowledge(channel_state,len(self.agents) + 1)#to tell which users got acknowledge for sending

        if(self.verbose):
            print ("channel states are: ",channel_state)
            for channel in channel_state:
                print("agents: {} transmiting on channel: {} ".format(channel,channel_state.index(channel)))
            print("ACK: {}".format(ack_arr))

        state =[]
        for i in range(2*self.number_of_channels + 1):
            state.append(0)
        state[action] = 1

        #state = #ack_arr
        info = {
                "channel state":channel_state,
                "acknowledge array":ack_arr
                }
        reward = self.reward(ack_arr, action)
        if reward==0.1:
            state.append(0)
        else:
            state.append(reward)
        done = not self.check_valid_action(action)  # done is an array  that represents a user in each index if index is true it means the user is trying to do a forbidden action
        print("state is: {}".format(state))
        return state,reward, done, info

    def render(self):
        pass

    def reset(self):
        state = []
        for i in range(2*self.number_of_channels + 2):
            state.append(0)
        return state



