from environment.generators.markov_generator import Markov_Generator
from environment.topologics.random_topology import simple_topology
"""
a general environment for training agents in a usecase of comunications networks
the api is very similar to a gym game environment.
"""
class environment(object):

    #constructor
    def __init__(self, verbose=False,num_of_users=3,num_of_channels=3,dqn = None,shape=None,path =None):#dqn and shape are needed for our dqn generator, other generators may not need to use them
        """
        this is the constractor of the environment object. here we create the users and the channels and the topology using a generator.
        we get the reward function from the topology. the generator can be changed to a different generator that will provide agents and channels and topology the fits
        the user needs, in those cases the user should import the new generator to the environment and change the constractor function of the environment to use the new
        generator.
        :param verbose: a variable that if is true when given will make the environment object to print more logs about each step
        :param num_of_users: an integer that represents the number of users the environment will host
        :param num_of_channels: an integer that represents the amount of channels the environment will host, an extra channel will be created and it will represent the action of not transmiting
        :param dqn: in some usecases the user might want the environment to hold a deep neural network especially in cases of training multiple agents at the same time, in this case he can give a refrence to the network in the dqn variable
        :param shape: the shape of dqn
        :param path: this is the path for the directory where the wieghts files will be for usecases in which the environment needs to update the wieghts of dqn
        """
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
        """
        this method incharge of synchronizing the environment accoridng to the channels that each user chose to transmit on at the current timestep
        :param action: a number that represent the channel a user chose to transmit on
        :return:
            state - the state of the user, shows where the user chose to transmit and what reward he got for that action for example
            [0,0,0,1,0,0,1.5] means that the user chose to transmit on channel 3 out of 5 (first channel represent not transmiting) and got a reward of 1.5 for that action
            reward - the reward the user got for this current time step
            done - a boolean that says if all users transimet legaly, if a user transmited in an ileageal way done will be true
            info - information about the current state of the environment, will return a dictionary with the users who chose to transmit on each chanel and also an array
            that shows which users got acknowledge for their action
        """
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



    def reset(self):
        """
        reset the environment. implementation may change for different usecases
        :return: a state of zeros
        """
        state = []
        for i in range(2*self.number_of_channels + 2):
            state.append(0)
        return state

    def after_episode(self):
        """
        an action the environment should do at the end of an episode,
        currently its only loading wieghts for dqn but may change for different usecases
        """
        self.dqn.load_weights(self.path)


    def create_states(self,actions,ack_arr):
        """
        sorts out the state of each user in the environment and triggers the agents after_step method at the end of each timestep
        :param actions: array that represents the action each user took
        :param ack_arr: array that represent the acknowledge each user got for his action in current timestep
        """
        for agent in self.agents:#dont want the first agent
            r = self.reward(ack_arr,actions[agent.id+1],agent.id+1)
            action = actions[agent.id+1]
            if(action == 0):
                r=0
            print("agent number :{} has reward of : {}".format(agent.id+1,r))
            agent.after_step(actions[agent.id+1],r)





