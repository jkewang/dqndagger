#add pretrain net for actor
#outline:
#  pretrained actor
#  actor-critic --> DDPG
#  human-control --> Data Aggregation & Policy Aggregation
#    Data Aggregation: add human control truple into replay buffer
#    Policy Aggregation:
#        traditional : P(i) = alpha*Pi(i-1) + (1-alpha)*Pi(*)
#        in our work : Loss = Loss(pi_theta(s) - pi_*(s)) + Loss(TD_err)
#    core idea: combine Data aggregation with policy aggregation in Reinforcement learning (DDPG) frame work
#

import tensorflow as tf
import numpy as np
import random
import math

class PretrainActor(object):
    def __init__(self):
        self.BATCH_SIZE = 32
        self.LR = 1e-5
    def read_data(self):

    def build_actor(self):

    def learn(self):

