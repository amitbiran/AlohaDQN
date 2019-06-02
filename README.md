# AlohaDQN
##### Table of Contents  
[Description](#description) 

[Structure](#structure)  

[Setup](#setup)

[Usage and Implementation](#usage)

<a name="description"/>

## Description
This repository is a part of our final project in Ben Gurion University. This project is in the field of deep reinforcment learning.
the project usecase is of N users who are comunicationg via a network with K channels, without sharing information.
meaning that the comunication is based on the slotted aloha protocol. each user can transmit on a channel or not transmit, if two users will transmit
on the same channel at the same timeslot there will be a collision and the transmission will fail.

<a name="structure"/>

## Structure
The repository is seprated into two parts:
- The first one is a learning environment. The environment was implemented in order to simulate our usecase in a generic way, means that if 
the structure of the network needs to change or the manner in which users are behaving in needs to change it is easy to implement.
The environment was implemented with qpi which is similar to the api of the GYM library. 
Although this environment was implemented to serve out final project we implemented it as a service,
we did put a lot of effort to make it as general as possible, so any usecase of users who are comunicating 
through multiple channel can use our environment.
- The second one is the training algorithm - we created a training algorithm that fits our usecase. our project focuses on the scenario of
network with two channels, and two users who transmit on those channels in probebility that depends on their last attempt in transmiting. In 
other words the users are behaving in a Markovian manner. out goal is to train a third agent to manage to share the two channels with those users
with as less collisions as possible.

<a name="setup"/>

## Setup
In order to use the training environment one should implement four objects that inherits from base objects inside the environment.
the implementation is simple and will require the user to implement one or two methods in order to make the environment fits their network usecase.
 - agent - there is a need to implement an agent object, this object will define the behaviour of the the agents (that are not being trained) in the network.
 ther is an option to implement and use multiple types of agents on the same environment.
 
 - topologic - this class define the manner in which we calculate rewards for each action an agent took.
 
 - generator - this class create users topologic and agents and provide them to the environment.
 
 - channel, a basic channel is implemented in case there is a need for a channel with complex functionality there will be need to implement a channel calss.
 
 once those four been implemented the environment is ready to use. 
 
 <a name="usage"/>
 
 ## Usage and Implementation
 #### using the environment in training
 ##### instantiation of the environment
```
    NUM_OF_CHANNELS = 2
    batch_size = 10
    number_of_steps = 3
    features = 2 * NUM_OF_CHANNELS + 2
    from environment.environment import environment
    env_shape = (batch_size,number_of_steps,features)
    env = environment(verbose=True, num_of_users=2, num_of_channels=NUM_OF_CHANNELS,dqn=env_network,shape=env_shape,path = path)
```
 the env_shape is needed only if the environment needs a neaural network. in our usecase it was not needed but we did gave the option.
 
  ##### taking a step
  ```
  state,reward,done,info=env.step(action)
  ```
  - action is a number that represent the number of channel the agent wish to transmit.
  - state - the state of the user, shows where the user chose to transmit and what reward he got for that action for example
            [0,0,0,1,0,0,1.5] means that the user chose to transmit on channel 3 out of 5 (first channel represent not transmiting) and got a reward of 1.5 for that action
  - reward - the reward the user got for this current time step
  - done - a boolean that says if all users transimet legaly, if a user transmited in an ileageal way done will be true
  - info - information about the current state of the environment, will return a dictionary with the users who chose to transmit on each chanel and also an array that shows which users got acknowledge for their action
  once a step been taken the environment will evaluate the new state after all actions of current timestep been taken and will return reward and new state
  ##### reset the environment
  ```
state = env.reset()
```

 #### The agent class
 ##### base agent class
 user of the environment should implement an agent that fits his usecase and inherit from this base agent.
 ```
 class Agent(object):

    def __init__(self,in_channel,out_channel,id,policy):
        self.in_channel = in_channel
        self.out_channel = out_channel
       # self.take_action = take_action
        self.id = id
        self.policy = policy

    def take_action(self,item):
        print (self.id)
        return self.id+1#self.policy.take_action(item)

    def get_in_channel(self):
        return self.in_channel

    def get_out_channel(self):
        return self.out_channel

    def after_step(self,state,reward):
        pass

 ```
 the agent behaviour is defined in the take_action method, the base agent will always transmit on the channel the match his id.
 the after_step method will be triggered by the environment at the end of the timestep to give the agent the ability to change his state according to the reward he recieved for his last action.
 
 ##### example for an agent that extends a base agent
 ```
 class MarkovAgent(Agent):
    def __init__(self,in_channel,out_channel,id,policy):
        Agent.__init__(self,in_channel,out_channel,id,policy)# call father constructor
        self.state = "success"
        self.succ_rate = 0.7
        self.fail_rate = 0.3
        self.action_taken = 0
        self.trans=0
    def take_action(self,item):
        trans = random.randint(1, 101)
        if(self.state == "success"):
            if(trans<=self.succ_rate*100):
                action = self.id+1
            else:
                action = 0
        else:
            if(trans<=self.fail_rate*100):
                action=self.id+1
            else:
                action =0
        self.action_taken = action
        self.trans = trans
        return action


    def after_step(self,action,reward):
        if(reward == 1):
            self.state ="success"
        else:
            self.state = "fail"
 ```
this is a Markovian agent, it will change his state depending on his success or failiure in his last attempt to transmit. different state means different probebility for transmission.
#### topology
##### base topology
a user who wishes to use the environment should inherit from base_topology and implement a topology with reward function for his usecase. 
the topology should be provided by the users generator
```
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

        return action >=0 and action <=len(self.channels)

    @abstractmethod
    def reward(self,arr):
        pass
```
the reward function will define the way the reward is given for a specific action an agent took.

##### example for a topology that inherit from base topology
```
class simple_topology(base_topology):
    punishment = -1
    last_0_reward = 0
    last_0_action = 0
    last_try = 0
    last_try_count = 1
    def reward(self,arr,action,id):
        if(id==0):

            if(action==0 and arr[1]==1 and arr[2]==1):
                self.last_0_reward = 0.5
                self.last_0_action = action
                self.punishment = -1
                return 0.5
            if(action == 0 and (arr[1] == 0 or arr[2] == 0)):
                self.last_0_reward = -0.5
                self.last_0_action = action
                self.punishment = -1
                return -0.5
            if(arr[0]==0):
                #to encourage agent not to stick to only one channel we will give him panelty if he tries the same channel again and again
                if(action == self.last_try):
                    self.last_try_count += 1
                else:
                    self.last_try_count = 1
                if(self.last_0_reward == -1 and self.last_0_action == action):
                    self.punishment *= 4
                    self.last_0_action = action
                    self.last_try = action
                    self.last_0_reward = -1
                    return self.punishment
                else:
                    self.punishment = -1
                    self.last_0_reward =-1
                    self.last_try = action
                    self.last_0_action = action
                    return -1
            if(arr[0]==1):
                #to encourage agent not to stick to only one channel we will give him panelty if he tries the same channel again and again
                self.punishment=-1
                self.last_0_action = action
                self.last_0_reward = 2
                if (action == self.last_try):
                    self.last_try_count += 1
                    return 2#0.2
                else:
                    self.last_try_count = 1
                    return 2
            return 0
        else:
            if(arr[id]==1):
                return 1
            else:
                return 0
```

#### generator
the generator class will be imported by the environment it define the way that the agents and the channels and the topology are being given to the environment.
for example if a user wish to implement multiple types of agents he should provide array with all different types of agents from the generator.
each generator provides the users
channels and topologics for the environment.

##### example of a simple generator
```
    def __init__(self,n_agents,n_channels):
        self.n_agents = n_agents
        self.n_channels = n_channels

    def generate_agents(self):
        rp = random_policy()
        agents_list = []
        for i in range(self.n_agents):
            agent_arr = []
            for j in range(self.n_channels):
                agent_arr.append(1)
            agents_list.append(Agent(agent_arr,agent_arr,i,random_policy()))
        return agents_list

    def generate_channels(self):
        channels_list = []
        for i in range(self.n_channels):
            channels_list.append(Channel(1))
        return channels_list


    def add_items(self,item1,item2):
        pass
```




