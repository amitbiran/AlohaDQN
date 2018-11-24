
class envorinment(object):

    #constructor
    def __init__(self, topology, verbose=False):
        self.verbose = verbose
        self.agents = topology.agents
        self.channels = topology.channels
        self.reward= topology.reward
        self.calculate_acknowledge = topology.calculate_acknowledge
        self.check_valid_action = topology.check_valid_action
        self.n = len(self.channels)


    def step(self, actions):
        done_n = [] #done is an array  that represents a user in each index if index is true it means the user is trying to do a forbidden action
        for i in range(len(actions)):
            done_n.append(self.check_valid_action(actions[i]))

        channel_state =[]#array of arrays, in each array is a list of all users tring to transmit on that channel.
        for cahnnel in self.channels:
            channel_state.append([])

        # iterate through the channels
        for i in range(self.n):
            #iterate through the actions per channel
            for j in range(len(actions)):
                if(actions[j][i]==1):#if someone tried to transmit on the channel add his id to the channel state array
                    channel_state[i].append(self.agents[j].id)


        ack_arr = self.calculate_acknowledge(channel_state,len(self.agents))#to tell which users got acknowledge for sending

        if(self.verbose):
            print ("channel states are: ",channel_state)
            for channel in channel_state:
                print("agents: {} transmiting on channel: {} ".format(channel,channel_state.index(channel)))
            print("ACK: {}".format(ack_arr))


        obs_n = ack_arr
        info_n = channel_state
        reward_n = self.reward(ack_arr)

        return obs_n,reward_n, done_n, info_n

    def render(self):
        pass

    def reset(self):
        pass



