from __future__ import division

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

from keras.models import Sequential, Model
from keras.layers import LSTM ,Input, Dense, Lambda

from keras.layers.convolutional import Conv2D
from keras.layers.merge import _Merge, Multiply
from keras import backend as K
NUM_OF_CHANNELS = 3
number_of_steps = 4#how many time steps are we training it for
batch_size = 10#how many examples we show model before updating weights
features = 2 * NUM_OF_CHANNELS + 2#the input is the size of 8 numbers for now todo
OUTPUT_DIM = NUM_OF_CHANNELS + 1
number_of_lstm_units = 100
h_size = int((number_of_steps/2)*number_of_lstm_units) #The size of the lstm layer before splitting it into Advantage and Value streams.

from gridworld import gameEnv
#env = gameEnv(partial=False,size=5)


env = environment(verbose=True,num_of_users=4, num_of_channels=NUM_OF_CHANNELS)
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
        A=9

    def get_q(self, state):
        return self.model.predict_on_batch(state)

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

"""
#train
#batch_size = 32 #How many experiences to use for each training step.todo
update_freq = 4 #How often to perform a training step.
y = .99 #Discount factor on the target Q-values
startE = 1 #Starting chance of random action
endE = 0.1 #Final chance of random action
annealing_steps = 10000. #How many steps of training to reduce startE to endE.
num_episodes = 100 #How many episodes of game environment to train network with.
pre_train_steps = 10000 #How many steps of random actions before training begins.
#max_epLength = 50 #The max allowed length of our episode.
load_model = False #Whether to load a saved model.
path = "./dqn" #The path to save our model to.

tau = 0.001 #Rate to update target network toward primary network

gamma = 0.9
start_eps = 1.
end_eps = 0.1
max_episode_length = 400
target_update_rate = 0.001

eps = start_eps
step_drop = (start_eps - end_eps) / annealing_steps

#store rewards and steps per episode
j_list = []
r_list = []
total_steps = 0

actor_network = Qnetwork(h_size)
target_network = Qnetwork(h_size)

experience = Experience(buffer_size=50000)

## Do this to periodically update the target network with ##
## the weights of the actor network                                   ##
# target_network.set_weights(actor_network.get_weights())


for i in range(num_episodes):
    print("start episode ",i)
    episode_exp = Experience(buffer_size=50000)
    s = env.reset()
    # s = resizeFrames(s)
    done = False
    total_reward = 0
    j = 0

    while j < max_episode_length:
        j += 1

        if np.random.rand(1) < eps or total_steps < pre_train_steps:
            a = np.random.randint(0, 4)
        else:
            prediction = actor_network.model.predict([s.reshape((1, 84, 84, 3)), np.zeros((32, 1))])
            a = np.argmax(prediction[0])

        s1, r, done,info = env.step(a)
        print("reward is:" , r)
        # s1 = resizeFrames(s1)
        total_steps += 1
        print(total_steps,pre_train_steps)
        episode_exp.storeExperience(np.reshape(np.array([s, a, r, s1, done]), [1, 5]))

        if total_steps > pre_train_steps:
            if eps > end_eps:
                eps -= step_drop

            if total_steps % update_freq == 0:
                train_batch = experience.sample(batch_size)
                print("hello world")
                # Have to do this because couldn't easily splice array
                # out of experience buffer, e.g.,
                # train_input = train_batch[:, 3]
                # when train_batch was a numpy array
                this_state = np.ndarray((batch_size,features))
                actions = np.ndarray((batch_size, 1))
                rewards = np.ndarray((batch_size, 1))
                next_state = np.ndarray((batch_size,features))
                dones = np.ndarray((batch_size, 1))
                for i in range(batch_size):
                    if(total_steps== 10004):
                        kkk=0
                        tt= train_batch[i][0]
                        ttt=train_batch[i][1]
                    ts = train_batch[i][0]
                    this_state[i] = ts
                    actions[i] = train_batch[i][1]
                    rewards[i] = train_batch[i][2]
                    next_state[i] = train_batch[i][3]
                    dones[i] = train_batch[i][4]

                q1 = actor_network.model.predict([next_state, np.zeros((32, 1))])
                q1 = np.argmax(q1[0], axis=3)

                q2 = target_network.model.predict([next_state, np.zeros((32, 1))])
                q2 = q2[0].reshape((batch_size, env.actions))

                end_multiplier = -(dones - 1)

                double_q = q2[range(32), q1.reshape((32))].reshape((32, 1))

                target_q = rewards + (gamma * double_q * end_multiplier)

                print("Target Q Shape: ", target_q.shape)

                q = actor_network.model.predict([this_state, actions])
                # q_of_actions = q[:, train_batch[:, 1]]
                print(target_q.shape)

                actor_network.model.fit([this_state, actions], [np.zeros((32, 1, 1, 4)), target_q])
                target_network.model.set_weights(actor_network.model.get_weights())

        total_reward += r
        s = s1

        if done == True:
            break

    experience.storeExperience(episode_exp.replay_buffer)
    j_list.append(j)
    r_list.append(total_reward)


print("Percent of succesful episodes: " + str(sum(r_list) / num_episodes) + "%")
"""
if __name__ == '__main__':
    agent = Qnetwork()
    state = np.asarray(env.reset())#size of feature
    zero_state = np.zeros(features)
    state_to_feed = [state]
    for i in range(number_of_steps-1):
        state_to_feed.append(zero_state)
    state_to_feed = np.asarray(state_to_feed)
    state_to_feed = np.reshape(state_to_feed,newshape=(1, number_of_steps, features))
    q_value_probability_distribuation = agent.get_q(state=state_to_feed)
    print (q_value_probability_distribuation)
    a = 5