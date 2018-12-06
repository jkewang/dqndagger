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
import math
import random
import numpy as np

class Actor_Critic(object):
    def __init__(self):
        self.BATCH_SIZE = 32
        self.LR_SA = 1e-3
        self.LR_A = 1e-3
        self.LR_C = 1e-3
        self.N_ACTIONS = 4
        self.N_OTHERS = 40
        self.N_SLIDING = 140
        self.tf_s_sliding = tf.placeholder(tf.float32, [None, self.N_SLIDING])
        self.tf_s_others = tf.placeholder(tf.float32, [None, self.N_OTHERS])
        self.tf_a = tf.placeholder(tf.float32, [None, 4])
        self.sess = tf.Session()
        self.learning_step = 0
        self.Saver = tf.train.Saver()

        self.pointer = 0
        self.memory = np.zeros((self.MEMORY_CAPACITY, (self.N_SLIDING+self.N_OTHERS) * 2 + self.N_ACTIONS + 1), dtype=np.float32)
        self.MEMORY_CAPACITY = 10000

        self.a = self.build_actor()
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.a, self.tf_a), axis=1))
        self.supervised_train_actor = tf.train.AdamOptimizer(self.LR_SA).minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())

    def build_actor(self,s):
        with tf.variable_scope("actor"):
            b1w = 0.01 * tf.Variable(tf.random_normal([self.N_SLIDING, 256], name="b1w"))
            b1b = 0.01 * tf.Variable(tf.zeros([1, 256]), name="b1b")
            b1lo = tf.matmul(self.tf_s_sliding, b1w) + b1b
            b1o = tf.nn.relu(b1lo)
            b2w = 0.01 * tf.Variable(tf.random_normal([256, 256]), name="b2w")
            b2b = 0.01 * tf.Variable(tf.zeros([1, 256]), name="b2b")
            b2lo = tf.matmul(b1o, b2w) + b2b
            b2o = tf.nn.relu(b2lo)
            tf_s_others = tf.reshape(self.tf_s_others, [-1, self.N_OTHERS])
            input_size3 = 256 + self.N_OTHERS
            output_size3 = 1024
            input_size4 = 1024
            output_size4 = 512
            input_size5 = 512
            output_size5 = 128
            input_size6 = 128
            output_size6 = self.N_ACTIONS
            b3w = 0.01 * tf.Variable(tf.random_normal([input_size3, output_size3]), name="b3w")
            b3b = tf.Variable(tf.zeros([1, output_size3]), name="b3b")
            myreal_input = tf.concat([b2o, tf_s_others], 1)
            b3lo = tf.matmul(myreal_input, b3w) + b3b
            b3o = tf.nn.relu(b3lo)
            b4w = 0.01 * tf.Variable(tf.random_normal([input_size4, output_size4]), name="b4w")
            b4b = tf.Variable(tf.zeros([1, output_size4]), name="b4b")
            b4lo = tf.matmul(b3o, b4w) + b4b
            b4o = tf.nn.relu(b4lo)
            b5w = 0.01 * tf.Variable(tf.random_normal([input_size5, output_size5]), name="b5w")
            b5b = tf.Variable(tf.zeros([1, output_size5]), name="b5b")
            b5lo = tf.matmul(b4o, b5w) + b5b
            b5o = tf.nn.relu(b5lo)
            b6w = 0.01 * tf.Variable(tf.random_normal([input_size6, output_size6]), name="b6w")
            b6b = tf.Variable(tf.zeros([1, output_size6]), name="b6b")
            self.a = tf.nn.softmax(tf.matmul(b5o,b6w)+b6b)
        return self.a

    def build_critic(self,s,a):
        with tf.variable_scope("critic"):
            b1w_s = 0.01 * tf.Variable(tf.random_normal([self.N_SLIDING, 256], name="b1w_s"))
            b1w_a = 0.01 * tf.Variable(tf.random_normal([self.N_ACTIONS, 256], name="b1w_a"))
            b1b = 0.01 * tf.Variable(tf.zeros([1, 256]), name="b1b_a")
            b1lo = tf.matmul(s , b1w_s) + tf.matmul(a, b1w_a) + b1b
            b1o = tf.nn.relu(b1lo)
            b2w = 0.01 * tf.Variable(tf.random_normal([256, 256]), name="b2w")
            b2b = 0.01 * tf.Variable(tf.zeros([1, 256]), name="b2b")
            b2lo = tf.matmul(b1o, b2w) + b2b
            b2o = tf.nn.relu(b2lo)
            tf_s_others = tf.reshape(self.tf_s_others, [-1, self.N_OTHERS])
            input_size3 = 256 + self.N_OTHERS
            output_size3 = 1024
            input_size4 = 1024
            output_size4 = 512
            input_size5 = 512
            output_size5 = 128
            input_size6 = 128
            output_size6 = self.N_ACTIONS
            b3w = 0.01 * tf.Variable(tf.random_normal([input_size3, output_size3]), name="b3w")
            b3b = tf.Variable(tf.zeros([1, output_size3]), name="b3b")
            myreal_input = tf.concat([b2o, tf_s_others], 1)
            b3lo = tf.matmul(myreal_input, b3w) + b3b
            b3o = tf.nn.relu(b3lo)
            b4w = 0.01 * tf.Variable(tf.random_normal([input_size4, output_size4]), name="b4w")
            b4b = tf.Variable(tf.zeros([1, output_size4]), name="b4b")
            b4lo = tf.matmul(b3o, b4w) + b4b
            b4o = tf.nn.relu(b4lo)
            b5w = 0.01 * tf.Variable(tf.random_normal([input_size5, output_size5]), name="b5w")
            b5b = tf.Variable(tf.zeros([1, output_size5]), name="b5b")
            b5lo = tf.matmul(b4o, b5w) + b5b
            b5o = tf.nn.relu(b5lo)
            b6w = 0.01 * tf.Variable(tf.random_normal([input_size6, output_size6]), name="b6w")
            b6b = tf.Variable(tf.zeros([1, output_size6]), name="b6b")
            self.q = tf.matmul(b5o,b6w)+b6b
        return self.q

    def store_transition(self,s,a,r,s_,done):
        transition = np.hstack((s,a,[r],s_))
        index = self.pointer % self.MEMORY_CAPACITY
        self.memory[index,:] = transition
        self.pointer += 1

    def choose_action(self,s):
        action = self.sess.run(self.a, {self.tf_s_sliding: s[], self.tf_s_others:s[]})[0]
        return action

    def learn(self):
        self.sess.run(self.soft_replace)

        indices = np.random.choice(self.MEMORY_CAPACITY, size=self.BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def saver(self):
        self.Saver.save(self.sess, './ac_model/model.ckpt')

