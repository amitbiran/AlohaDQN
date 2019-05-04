from generators.simple_generator import Simple_Generator
from generators.markov_generator import Markov_Generator
from generators.dqn_generator import Dqn_Generator
from topologics.random_topology import simple_topology
from Qnetwork import Qnetwork
class environment(object):

    #constructor
    def __init__(self, verbose=False,num_of_users=3,num_of_channels=3,dqn = None,shape=None,path =None):#dqn and shape are needed for our dqn generator, other generators may not need to use them
        self.verbose = verbose
        generator = Markov_Generator(n_agents=num_of_users,n_channels=num_of_channels)
        #generator.add_items(dqn,shape)
        self.path = path
        self.dqn =dqn
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
            for i in range(len(channel_state)):
                print("agents: {} transmiting on channel: {} ".format(channel_state[i],i))
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
        reward = self.reward(ack_arr, action,0)
        if reward==0.1:
            state.append(0)
        else:
            state.append(reward)
        done = not self.check_valid_action(action)  # done is an array  that represents a user in each index if index is true it means the user is trying to do a forbidden action
        print("state is: {}".format(state))

        self.create_states(actions,ack_arr)
        return state,reward, done, info

    def render(self):
        pass

    def reset(self):
        state = []
        for i in range(2*self.number_of_channels + 2):
            state.append(0)
        return state

    def after_episode(self):
        self.dqn.load_weights(self.path)


    def create_states(self,actions,ack_arr):
        #print("agent number: 0 has reward of {}".format(self.reward(ack_arr,actions[0],0)))
        for agent in self.agents:#dont want the first agent
            r = self.reward(ack_arr,actions[agent.id+1],agent.id+1)
            action = actions[agent.id+1]
            if(action == 0):
                r=0
            print("agent number :{} has reward of : {}".format(agent.id+1,r))
            agent.after_step(actions[agent.id+1],r)





