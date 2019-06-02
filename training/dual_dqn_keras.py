from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from environment.environment import environment
from environment.buffers.states_buffer import StatesBuffer
from environment.buffers.StepsBuffer import StepsBuffer
from environment.buffers.EpisodesBuffer import EpisodesBuffer
from training.Qnetwork import Qnetwork
import datetime
import os

countt=0
NUM_OF_CHANNELS = 2
number_of_steps = 3#how many time steps are we training it for
batch_size = 10#how many examples we show model before updating weights
features = 2 * NUM_OF_CHANNELS + 2
OUTPUT_DIM = NUM_OF_CHANNELS + 1
number_of_lstm_units = 100
h_size = int((number_of_steps/2)*number_of_lstm_units) #The size of the lstm layer before splitting it into Advantage and Value streams.


#env = gameEnv(partial=False,size=5)
#implement network


if __name__ == '__main__':
    timestamp = str(datetime.datetime.now())  # .split('.')
    timestamp = timestamp.replace(':', '-')
    path = os.path.join(os.getcwd(), 'dqn')
    path = os.path.join(path, timestamp) + " - model"
    os.makedirs(path)
    print("created folder for model at {}".format(path))

    print("writing general variables")
    with open(os.path.join(path, "general_variables.txt"), "w+") as f:
        f.write("NUM_OF_CHANNELS = {}\n".format(NUM_OF_CHANNELS))
        f.write("number_of_steps = {}\n".format(number_of_steps))
        f.write("batch_size = {}\n".format(batch_size))
        f.write("features = {}\n".format(features))
        f.write("OUTPUT_DIM = {}\n".format(OUTPUT_DIM))
        f.write("number_of_lstm_units = {}\n".format(number_of_lstm_units))
        f.write("h_size = {}\n".format(h_size))

    #create env
    env_network = Qnetwork(number_of_steps=number_of_steps,
                             features=features,
                             number_of_lstm_units=number_of_lstm_units,
                             OUTPUT_DIM=OUTPUT_DIM)
    env_shape = (batch_size,number_of_steps,features)
    env = environment(verbose=True, num_of_users=2, num_of_channels=NUM_OF_CHANNELS,dqn=env_network,shape=env_shape,path = path)

    #general params
    update_freq = 4  # How often to perform a training step.
    y = .99  # Discount factor on the target Q-values
    startE = 1  # Starting chance of random action
    endE = 0.1  # Final chance of random action
    annealing_steps = 10000.  # How many steps of training to reduce startE to endE.
    num_episodes = 10 # How many episodes of game environment to train network with.
    pre_train_steps = 1000  # How many steps of random actions before training begins.
    load_model = False  # Whether to load a saved model.
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
    succ_rate_list = []
    fail_rate_list = []
    sent_rate_list =[]
    succ_out_of_sent_list = []
    collide_count_list =[]
    myBuffer = EpisodesBuffer()
    total_steps = 0

    #create DQNs
    actor_network = Qnetwork(number_of_steps=number_of_steps,
                             features=features,
                             number_of_lstm_units=number_of_lstm_units,
                             OUTPUT_DIM=OUTPUT_DIM)
    target_network = Qnetwork(number_of_steps=number_of_steps,
                             features=features,
                             number_of_lstm_units=number_of_lstm_units,
                             OUTPUT_DIM=OUTPUT_DIM)

    actor_network.save_model(path)
    actor_network.save_weights(path)
    #training loop
    for i in range(num_episodes):
        print("i: ", i)
        # each episode we reset the env and variables
        steps_buffer = StepsBuffer(number_of_steps=number_of_steps)
        state = env.reset()
        suc_count = 0
        sent_count =0
        collide_count =0
        gave_up_non_available_count =0
        gave_up_is_available_count =0
        fail_count = 0
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
            steps_buffer.add(np.reshape(np.array([state, action, reward, state1, done]), [1, 5]))  # Save the experience to our episode buffer.
            if(total_steps>pre_train_steps):
                #normalize eps
                if(eps>end_eps):
                    eps-=step_drop
                if(total_steps%update_freq==0):#if we need to update
                    train_batch = myBuffer.sample(batch_size)#Get a random batch of experiences.

                    # Below we perform the Double-DQN update to the target Q-values
                    this_state = np.ndarray((batch_size,number_of_steps, features))
                    actions = np.ndarray((batch_size,number_of_steps))
                    rewards = np.ndarray((batch_size, number_of_steps))

                    next_state = np.ndarray((batch_size,number_of_steps, features))
                    dones = np.ndarray((batch_size,1))
                    for ii in range(batch_size):
                        this_state[ii] = train_batch[ii][0]
                        actions[ii] = train_batch[ii][5]


                        #tem.append(train_batch[ii][2])
                        rewards[ii] = np.array(train_batch[ii][6])
                        next_state[ii] = train_batch[ii][3]
                        dones[ii] = train_batch[ii][4]


                    #get q values from networks
                    q1=actor_network.get_q(next_state)
                    q1=np.argmax(q1[0])

                    q2 = target_network.get_q(next_state)#according to dor this is the next q value
                    q2=q2[0]

                    target_batch = target_network.get_q(this_state)


                    end_multiplier = 1
                    double_q = q2[q1]
                    target_q = rewards + (gamma*double_q*end_multiplier)
                   # print(target_q.shape)


                    for index in range(batch_size):
                       # r = train_batch[index][2]
                        target = train_batch[index][2]+(gamma*double_q*end_multiplier)
                        target_batch[index][train_batch[index][1]]=target#train_batch[index][1] this is the action itself exmp(0 or 3 or 2)

                    q=actor_network.get_q(this_state)#feed state to main network
                    print("value of q is: ", q)

                    actor_network.model.fit(this_state,target_batch)
                    print("fit done: ",countt)
                    countt+=1
            state=state1
            states_buffer.add(state)
            total_reward+=reward
            if(reward <= -1):
                fail_count +=1
                collide_count +=1
                sent_count +=1
            if(reward == 2 ):
                suc_count+=1
                sent_count +=1
            if(reward == -0.5):
                gave_up_is_available_count +=1
            if(reward == 0.5):

                gave_up_non_available_count +=1
            try_count+=1

            with open(os.path.join(path, "actions taken.txt"), "a+") as f:
                f.read()
                f.write("{} transmited on {} reward is {}\n".format(info["channel state"],action,reward))


        print("success {} out of {} -> {}".format(str(suc_count),str(try_count),str(suc_count/try_count)))
        print("success {} out of sent {}".format(str(suc_count),str(sent_count)))
        sent_rate_list.append(sent_count/try_count)
        succ_out_of_sent_list.append(suc_count/sent_count)
        collide_count_list.append(fail_count/sent_count)
        succ_rate_list.append(suc_count/try_count)
        fail_rate_list.append(fail_count / try_count)
        myBuffer.add(steps_buffer.buffer)
        actor_network.save_weights(path)
        target_network.load_weights(path)
        #env.after_episode()
        j_list.append(j)
        r_list.append(total_reward)

#print("Percent of succesful episodes: " + str(sum(r_list)/num_episodes) + "%")
print(succ_rate_list)
#print plots
plt.plot(succ_rate_list)
plt.ylabel('success rate in transmiting on a channel')
plt.savefig(os.path.join(path, 'success.png'))
plt.show()

plt.plot(fail_rate_list)
plt.ylabel('fail rate in transmiting on a channel')
plt.savefig(os.path.join(path, 'fail.png'))
plt.show()

plt.plot(sent_rate_list)
plt.ylabel('sent rate out of steps')
plt.savefig(os.path.join(path, 'sent_rate.png'))
plt.show()

plt.plot(succ_out_of_sent_list)
plt.ylabel('success rate in transmiting on a channel out of all sent tries')
plt.savefig(os.path.join(path, 'succ_out_of_sent.png'))
plt.show()

plt.plot(collide_count_list)
plt.ylabel('collide count rate out of sent')
plt.savefig(os.path.join(path, 'collide rate.png'))
plt.show()





# rMat = np.resize(np.array(r_list),[len(r_list)//100,100])
# rMean = np.average(rMat,1)
# plt.plot(rMean)