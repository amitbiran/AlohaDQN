from __future__ import division
#refrence https://arxiv.org/pdf/1509.06461.pdf
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import scipy.misc
import os
from tensorflow.contrib import rnn
from tensorflow.python.ops import math_ops
#%matplotlib inline
from environment.environment import environment
from states_buffer import StatesBuffer
from keras.models import Sequential, Model
from keras.layers import LSTM ,Input, Dense, Lambda

from keras.layers.convolutional import Conv2D
from keras.layers.merge import _Merge, Multiply
from keras import backend as K

countt=0
NUM_OF_CHANNELS = 3
number_of_steps = 4#how many time steps are we training it for
batch_size = 10#how many examples we show model before updating weights
features = 2 * NUM_OF_CHANNELS + 2#the input is the size of 8 numbers for now todo
OUTPUT_DIM = NUM_OF_CHANNELS + 1
number_of_lstm_units = 100
h_size = int((number_of_steps/2)*number_of_lstm_units) #The size of the lstm layer before splitting it into Advantage and Value streams.



from gridworld import gameEnv
#env = gameEnv(partial=False,size=5)
#implement network

class QLayer(_Merge):
    '''Q Layer that merges an advantage and value layer'''
    def _merge_function(self, inputs):
        '''Assume that the inputs come in as [value, advantage]'''
        output = inputs[0] + (inputs[1] - K.mean(inputs[1], axis=1, keepdims=True))
        return output


class Qnetwork():
    def __init__(self):

        self.state = Input(shape=(number_of_steps, features), dtype=tf.float32)
        LSTM_layer = LSTM(units=number_of_lstm_units, return_sequences=False)(self.state)#make it with an abstract batch_size

       # Splice outputs of lastm layer using lambda layer
        x_value = Lambda(lambda LSTM_layer: LSTM_layer[ :, :number_of_lstm_units // 2])(LSTM_layer)
        x_advantage = Lambda(lambda LSTM_layer: LSTM_layer[:, number_of_lstm_units // 2:])(LSTM_layer)

       # Process spliced data stream into value and advantage function
        value = Dense(activation="linear", units=10)(x_value)
        advantage = Dense(activation="linear", units=10)(x_advantage)

       # self.actions = Input(shape=(1,), dtype='int32')
        #self.actions_onehot = Lambda(K.one_hot, arguments={'num_classes': 10}, output_shape=(None,10))(self.actions)#todo instead of 10 was env.actions

        q = QLayer()([value, advantage])  # output dim is now 10
        self.q_out = Dense(activation="softmax", units=OUTPUT_DIM)(q)
        #self.Q_out = Multiply()([q, self.actions_onehot])
        #self.Q_out = Lambda(lambda x: K.cumsum(x, axis=3), output_shape=(1,))(self.Q_out)



        self.model = Model(inputs=[self.state], outputs=[self.q_out])
        self.model.compile(optimizer="Adam", loss="mean_squared_error")
        # # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        # self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        # self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        # self.actions_onehot = tf.one_hot(self.actions, env.actions, dtype=tf.float32)
        #
        # self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        #
        # self.td_error = tf.square(self.targetQ - self.Q)
        # self.loss = tf.reduce_mean(self.td_error)
        # self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        # self.updateModel = self.trainer.minimize(self.loss)
        self.model.summary()

    def get_q(self, state):
        o=self.model.predict_on_batch(state)
        return o

class Experience():

    def __init__(self, buffer_size):
        self.replay_buffer = []
        self.buffer_size = buffer_size

    def storeExperience(self, exp):
        if (len(exp) + self.buffer_size >= len(self.replay_buffer)):
            del self.replay_buffer[:(len(exp) + len(self.replay_buffer) - self.buffer_size)]

        self.replay_buffer.extend(exp)

        return self.replay_buffer

    def sample(self, sample_size):
        # return np.reshape(np.array(random.sample(self.replay_buffer, sample_size)), [sample_size, env_size])
        return random.sample(self.replay_buffer, sample_size)
#functions
class experience_buffer():
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])

def processState(states):
    #return np.reshape(states,[21168])
    return states

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        va = var.value()
        vi = tfVars[idx+total_vars//2].value()
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)




class experience_buffer():
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size


    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])

if __name__ == '__main__':
    #create env
    env = environment(verbose=True, num_of_users=4, num_of_channels=NUM_OF_CHANNELS)



    # agent = Qnetwork()
    # state = np.asarray(env.reset())#size of feature
    # zero_state = np.zeros(features)
    # state_to_feed = [state]
    # for i in range(number_of_steps-1):
    #     state_to_feed.append(zero_state)
    # state_to_feed = np.asarray(state_to_feed)
    # state_to_feed = np.reshape(state_to_feed,newshape=(1, number_of_steps, features))
    # q_value_probability_distribuation = agent.get_q(state=state_to_feed)
    # print (q_value_probability_distribuation)
    # a = 5

    update_freq = 4  # How often to perform a training step.
    y = .99  # Discount factor on the target Q-values
    startE = 1  # Starting chance of random action
    endE = 0.1  # Final chance of random action
    annealing_steps = 10000.  # How many steps of training to reduce startE to endE.
    num_episodes = 100  # How many episodes of game environment to train network with.
    pre_train_steps = 1000  # How many steps of random actions before training begins.
    # max_epLength = 50 #The max allowed length of our episode.
    load_model = False  # Whether to load a saved model.
    path = "./dqn"  # The path to save our model to.

    tau = 0.001  # Rate to update target network toward primary network

    gamma = 0.9
    start_eps = 1.
    end_eps = 0.1
    max_episode_length = 400
    target_update_rate = 0.001

    eps = start_eps
    step_drop = (start_eps - end_eps) / annealing_steps



    #create buffers
    j_list = []
    r_list = []

    myBuffer = experience_buffer()
    total_steps = 0

    actor_network = Qnetwork()
    target_network = Qnetwork()

    for i in range(num_episodes):
        print("i: ", i)
        # each episode we reset the env and variables
        episodeBuffer = experience_buffer()
        state = env.reset()
        suc_count = 0
        try_count = 0
        done = False
        total_reward = 0
        j = 0
        states_buffer = StatesBuffer(number_of_steps)


        while j<max_episode_length:
            print("j: ",j)
            #here is what we do each episode
            j+=1
            if np.random.rand(1) < eps or total_steps < pre_train_steps:
                action = np.random.randint(0, NUM_OF_CHANNELS+1)#before we gained enough experience we choose action randomly
            else:
                #if we have enough experience we feed the data to the network and get a predicition

                buff = states_buffer.buff
                if (len(buff)<number_of_steps):
                    buff = np.zeros((number_of_steps,features))
                state_for_input = np.array([buff])

                prediction = actor_network.get_q(state_for_input)
                action=np.argmax(prediction[0])

            state1,reward,done,info=env.step(action)#take a step with the action that was chosen
            print("reward is: ",reward)
            total_steps+=1#so we know we took a step
            episodeBuffer.add(np.reshape(np.array([state, action, reward, state1, done]), [1, 5]))  # Save the experience to our episode buffer.
            if(total_steps>pre_train_steps):
                #normalize eps
                if(eps>end_eps):
                    eps-=step_drop
                if(total_steps%update_freq==0):#if we need to update
                    train_batch = myBuffer.sample(batch_size)#Get a random batch of experiences.

                    # Below we perform the Double-DQN update to the target Q-values
                    this_state = np.ndarray((batch_size,number_of_steps, features))
                    actions = np.ndarray((batch_size,1))
                    rewards = np.ndarray((batch_size, 4))
                    next_state = np.ndarray((batch_size,number_of_steps, features))
                    dones = np.ndarray((batch_size,1))
                    for ii in range(batch_size):
                        this_state[ii] = train_batch[ii][0]
                        actions[ii] = train_batch[ii][1]
                        tem=[]#todo fix actions to give it properly not clones
                        for jj in range(number_of_steps):
                            tem.append(train_batch[ii][2])
                        rewards[ii] = np.array(tem)
                        next_state[ii] = train_batch[ii][3]
                        dones[ii] = train_batch[ii][4]


                    #get q values from networks
                    q1=actor_network.get_q(next_state)
                    q1=np.argmax(q1[0])

                    q2 = target_network.get_q(next_state)
                    q2=q2[0]

                    end_multiplier = -(dones - 1)
                    double_q = q2[q1]
                    target_q = rewards + (gamma*double_q*end_multiplier)
                   # print(target_q.shape)

                    q=actor_network.get_q(this_state)#feed state to main network
                    print("value of q is: ", q)
                    #orgenize input
                    #todo fix the arr to input so it gives it properly and not clones
                    arr_to_input = []#np.array()
                    for a in actions:
                        arr_for_one_action=np.zeros((number_of_steps, features))
                        action1 = np.zeros(features)
                        action1[int(a)]=1
                        for k in range(number_of_steps):
                            arr_for_one_action[k]=action1
                        arr_to_input.append(arr_for_one_action)
                    arr_to_input= np.array(arr_to_input)




                    actor_network.model.fit(arr_to_input,target_q)
                    print("fit done: ",countt)
                    countt+=1
            state=state1
            states_buffer.add(state)
            total_reward+=reward
            if(reward == 1):
                suc_count+=1
            try_count+=1

        print("success {} out of {} -> {}".format(str(suc_count),str(try_count),str(suc_count/try_count)))
        myBuffer.add(episodeBuffer.buffer)
        j_list.append(j)
        r_list.append(total_reward)
print("Percent of succesful episodes: " + str(sum(r_list)/num_episodes) + "%")

rMat = np.resize(np.array(r_list),[len(r_list)//100,100])
rMean = np.average(rMat,1)
plt.plot(rMean)