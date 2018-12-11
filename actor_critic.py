#add  net for actor
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
        self.r_list = []
        self.q_list = []

        self.EPSILON = 0.1
        self.BATCH_SIZE = 32
        self.TAU = 0.01
        self.GAMMA = 0.9
        self.LR_SA = 1e-3
        self.LR_A = 1e-3
        self.LR_C = 1e-3
        self.N_ACTIONS = 4
        self.N_OTHERS = 40
        self.N_SLIDING = 140
        self.a_replace_counter, self.c_replace_counter = 0,0
        self.tf_s_sliding = tf.placeholder(tf.float32, [None, self.N_SLIDING])
        self.tf_s_others = tf.placeholder(tf.float32,[None,self.N_OTHERS])
        self.tf_s_sliding_ = tf.placeholder(tf.float32, [None, self.N_SLIDING])
        self.tf_s_others_ = tf.placeholder(tf.float32,[None, self.N_OTHERS])
        self.tf_a = tf.placeholder(tf.float32, [None, 4])
        self.R = tf.placeholder(tf.float32,[None, 1])
        self.sess = tf.Session()
        self.learning_step = 0

        self.MEMORY_CAPACITY = 1000
        self.pointer = 0
        self.memory = np.zeros((self.MEMORY_CAPACITY, (self.N_SLIDING+self.N_OTHERS) * 2 + 1 + 1), dtype=np.float32)

        with tf.variable_scope('actor'):
            self.a = self.build_actor(self.tf_s_sliding,self.tf_s_others, scope='eval', trainable=True)
            a_ = self.build_actor(self.tf_s_sliding_,self.tf_s_others_, scope='target', trainable=False)
        with tf.variable_scope('critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            self.q = self.build_critic(self.tf_s_sliding,self.tf_s_others, self.a, scope='eval', trainable=True)
            self.q_ = self.build_critic(self.tf_s_sliding_,self.tf_s_others_,a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - self.TAU) * ta + self.TAU * ea), tf.assign(tc, (1 - self.TAU) * tc + self.TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        self.q_target = self.R + self.GAMMA * self.q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        self.td_error = tf.losses.mean_squared_error(labels=self.q_target, predictions=self.q)
        self.ctrain = tf.train.AdamOptimizer(self.LR_C).minimize(self.td_error, var_list=self.ce_params)

        self.a_loss = - tf.reduce_mean(self.q)  # maximize the q
        self.a_loss_supervised = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.a, self.tf_a), axis=1))

        self.atrain_reinforce = tf.train.AdamOptimizer(self.LR_A).minimize(self.a_loss, var_list=self.ae_params)
        self.atrain_supervised = tf.train.AdamOptimizer(self.LR_SA).minimize(self.a_loss_supervised,var_list=self.ae_params)

        self.init_all_weights("./model/model.ckpt",self.sess)
        #self.sess.run(tf.global_variables_initializer())

    def build_actor(self,tf_s_sliding,tf_s_others,scope,trainable):
        with tf.variable_scope(scope):
            b1w = 0.01 * tf.Variable(tf.random_normal([self.N_SLIDING, 256]), name="b1w",trainable=trainable)
            b1b = 0.01 * tf.Variable(tf.zeros([1, 256]), name="b1b",trainable=trainable)
            b1lo = tf.matmul(tf_s_sliding, b1w) + b1b
            b1o = tf.nn.relu(b1lo)
            b2w = 0.01 * tf.Variable(tf.random_normal([256, 256]), name="b2w",trainable=trainable)
            b2b = 0.01 * tf.Variable(tf.zeros([1, 256]), name="b2b",trainable=trainable)
            b2lo = tf.matmul(b1o, b2w) + b2b
            b2o = tf.nn.relu(b2lo)
            tf_s_others = tf.reshape(tf_s_others, [-1, self.N_OTHERS])
            input_size3 = 256 + self.N_OTHERS
            output_size3 = 1024
            input_size4 = 1024
            output_size4 = 512
            input_size5 = 512
            output_size5 = 128
            input_size6 = 128
            output_size6 = self.N_ACTIONS
            b3w = 0.01 * tf.Variable(tf.random_normal([input_size3, output_size3]), name="b3w",trainable=trainable)
            b3b = tf.Variable(tf.zeros([1, output_size3]), name="b3b",trainable=trainable)
            myreal_input = tf.concat([b2o, tf_s_others], 1)
            b3lo = tf.matmul(myreal_input, b3w) + b3b
            b3o = tf.nn.relu(b3lo)
            b4w = 0.01 * tf.Variable(tf.random_normal([input_size4, output_size4]), name="b4w",trainable=trainable)
            b4b = tf.Variable(tf.zeros([1, output_size4]), name="b4b",trainable=trainable)
            b4lo = tf.matmul(b3o, b4w) + b4b
            b4o = tf.nn.relu(b4lo)
            b5w = 0.01 * tf.Variable(tf.random_normal([input_size5, output_size5]), name="b5w",trainable=trainable)
            b5b = tf.Variable(tf.zeros([1, output_size5]), name="b5b",trainable=trainable)
            b5lo = tf.matmul(b4o, b5w) + b5b
            b5o = tf.nn.relu(b5lo)
            b6w = 0.01 * tf.Variable(tf.random_normal([input_size6, output_size6]), name="b6w",trainable=trainable)
            b6b = tf.Variable(tf.zeros([1, output_size6]), name="b6b",trainable=trainable)
            a = tf.nn.softmax(tf.matmul(b5o,b6w)+b6b)
        return a

    def build_critic(self,tf_s_sliding,tf_s_others,a,scope,trainable):
        with tf.variable_scope(scope):
            b1w = 0.01 * tf.Variable(tf.random_normal([self.N_SLIDING, 256]), name="b1w",trainable=trainable)
            b1b = 0.01 * tf.Variable(tf.zeros([1, 256]), name="b1b_a",trainable=trainable)
            b1lo = tf.matmul(tf_s_sliding , b1w) + b1b
            b1o = tf.nn.relu(b1lo)
            b2w = 0.01 * tf.Variable(tf.random_normal([256, 256]), name="b2w",trainable=trainable)
            b2b = 0.01 * tf.Variable(tf.zeros([1, 256]), name="b2b",trainable=trainable)
            b2lo = tf.matmul(b1o, b2w) + b2b
            b2o = tf.nn.relu(b2lo)
            tf_s_others = tf.reshape(tf_s_others, [-1, self.N_OTHERS])
            input_size3 = 256 + self.N_OTHERS
            output_size3 = 1024
            input_size4 = 1024
            output_size4 = 512
            input_size5 = 512
            output_size5 = 128
            input_size6 = 128
            output_size6 = self.N_ACTIONS

            b3w_s = 0.01 * tf.Variable(tf.random_normal([input_size3, output_size3]), name="b3w_s",trainable=trainable)
            b3w_a = 0.01 * tf.Variable(tf.random_normal([self.N_ACTIONS, output_size3]), name="b3w_a",trainable=trainable)
            b3b = tf.Variable(tf.zeros([1, output_size3]), name="b3b",trainable=trainable)
            myreal_input = tf.concat([b2o, tf_s_others], 1)
            b3lo = tf.matmul(myreal_input, b3w_s) + tf.matmul(a,b3w_a) + b3b
            b3o = tf.nn.relu(b3lo)
            b4w = 0.01 * tf.Variable(tf.random_normal([input_size4, output_size4]), name="b4w",trainable=trainable)
            b4b = tf.Variable(tf.zeros([1, output_size4]), name="b4b",trainable=trainable)
            b4lo = tf.matmul(b3o, b4w) + b4b
            b4o = tf.nn.relu(b4lo)
            b5w = 0.01 * tf.Variable(tf.random_normal([input_size5, output_size5]), name="b5w",trainable=trainable)
            b5b = tf.Variable(tf.zeros([1, output_size5]), name="b5b",trainable=trainable)
            b5lo = tf.matmul(b4o, b5w) + b5b
            b5o = tf.nn.relu(b5lo)
            b6w = 0.01 * tf.Variable(tf.random_normal([input_size6, output_size6]), name="b6w",trainable=trainable)
            b6b = tf.Variable(tf.zeros([1, output_size6]), name="b6b",trainable=trainable)
            q = tf.matmul(b5o,b6w)+b6b
        return q

    def store_transition(self,s_sliding,s_others,a,r,s_sliding_,s_others_,done):
        transition = np.hstack((s_sliding,s_others,a,[r],s_sliding_,s_others_))
        index = self.pointer % self.MEMORY_CAPACITY
        self.memory[index,:] = transition
        self.pointer += 1

    def choose_action(self,s):
        s_sliding = s[0][np.newaxis, :]
        s_others = s[1][np.newaxis, :]

        if np.random.uniform() < self.EPSILON:
            action = np.argmax(self.sess.run(self.a, {self.tf_s_sliding: s_sliding, self.tf_s_others: s_others})[0])

        else:
            action = np.random.randint(0, self.N_ACTIONS)
        return action

    def learn(self):
        self.sess.run(self.soft_replace)

        indices = np.random.choice(self.MEMORY_CAPACITY, size=self.BATCH_SIZE)
        bt = self.memory[indices, :]
        bs_sliding = bt[:, :self.N_SLIDING]
        bs_others = bt[:, self.N_SLIDING:(self.N_SLIDING+self.N_OTHERS)]
        ba = bt[:, (self.N_SLIDING+self.N_OTHERS): (self.N_SLIDING+self.N_OTHERS)+self.N_ACTIONS]
        br = bt[:, -(self.N_SLIDING+self.N_OTHERS) - 1: -(self.N_SLIDING+self.N_OTHERS)]
        bs_sliding_ = bt[:,-(self.N_SLIDING+self.N_OTHERS):-self.N_OTHERS]
        bs_others_ = bt[:, -self.N_OTHERS:]

        self.sess.run(self.atrain_reinforce, {self.tf_s_sliding: bs_sliding,self.tf_s_others:bs_others})
        self.sess.run(self.ctrain, {self.tf_s_sliding: bs_sliding,self.tf_s_others:bs_others, self.a: ba, self.R: br, self.tf_s_sliding_: bs_sliding_,self.tf_s_others_:bs_others_})

        self.sess.run(self.atrain_supervised,{self.tf_s_sliding:bs_sliding,self.tf_s_others:bs_others,self.tf_a:ba})
        print("learning")

    def init_all_weights(self, weight_file, sess):
        variables = tf.contrib.framework.get_variables_to_restore()
        for v in variables:
            print(v.name)
        variables_to_restore = [v for v in variables if (v.name[0] == 'a' and (v.name.find('Adam')) == -1)]
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(sess, weight_file)

        sess.run(tf.variables_initializer(self.get_uninitialized_variables(sess)))

    def get_uninitialized_variables(self, sess):
        global_vars = tf.global_variables()
        # print([str(i.name) for i in global_vars])
        is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
        print([str(i.name) for i in not_initialized_vars])
        return not_initialized_vars

    def saver(self):
        tf.train.Saver().save(self.sess, './ac_model/model.ckpt')

