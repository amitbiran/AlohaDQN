#!/usr/bin/env python
import os,sys
from topologics.random_topology import simple_topology
from environment.environment import envorinment


if __name__ == '__main__':
    number_of_episodes = 100
    number_of_timeslots = 10





    #create multiagent environment
    topology = simple_topology(agents,channels)
    env = envorinment(topology,True)

    for i in range(number_of_episodes):
        time_slot = 0
        while time_slot<number_of_timeslots:
            act_n = []
            for agent in topology.agents:
                act_n.append(agent.take_action(len(channels)))
            obs_n, reward_n, done_n, info_n = env.step(act_n)
            time_slot+=1
        env.reset()
        print("end of episode: {}".format(i))


