import tensorflow as tf
import numpy as np
import random

class Test(object):
    def __init__(self):
        self.BATCH_SIZE = 32
        self.LR = 1e-2
        self.N_ACTIONS = 4
        self.N_OTHERS = 40
        self.N_SLIDING = 140
        self.tf_s_sliding = tf.placeholder(tf.float32, [None, self.N_SLIDING])
        self.tf_s_others = tf.placeholder(tf.float32, [None, self.N_OTHERS])
        self.tf_a = tf.placeholder(tf.float32, [None, 4])
        self.sess = tf.Session()
        self.learning_step = 0

        self.filename = "pretrained_data.txt"
        self.f = open(self.filename)
        self.MAXLINE = 426607
        self.OccState = []
        self.VehState = []
        self.action = []

        self.a, self.a2 = self.build_actor()
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.a, self.tf_a), axis=1))
        self.loss2 = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.a2, self.tf_a), axis=1))
        self.train1 = tf.train.AdamOptimizer(self.LR).minimize(self.loss)
        self.train2 = tf.train.AdamOptimizer(self.LR).minimize(self.loss2)
        self.load_model()

    def load_model(self):
        saver = tf.train.Saver()
        saver.restore(self.sess,"./model/model.ckpt")

    def read_test_data(self):
        pointer = 0
        subdata = " "

        while (pointer < self.MAXLINE):
            subdata += self.f.readline()
            pointer += 1

        singledata = str.split(subdata, '#')
        for single in singledata:
            Occ_i = []
            Veh_i = []

            first_mark = single.rfind("!")
            second_mark = single.rfind("@")
            OccStr = single[1:first_mark - 1]
            VehStr = single[first_mark + 2:second_mark - 1]
            Action = single[second_mark + 1:]

            OccStr = OccStr.replace("\n", "")
            VehStr = VehStr.replace("\n", "")
            OccSpecific = OccStr.split(" ")

            print(OccSpecific)
            for item in OccSpecific:
                Occ_i.append(float(item))

            VehSpecific = VehStr.split(" ")
            for item in VehSpecific:
                if item != '':
                    Veh_i.append(float(item))

            self.OccState.append(Occ_i)
            self.VehState.append(Veh_i)
            if Action == '0':
                self.action.append([1.0, 0.0, 0.0, 0.0])
            elif Action == '1':
                self.action.append([0.0, 1.0, 0.0, 0.0])
            elif Action == '2':
                self.action.append([0.0, 0.0, 1.0, 0.0])
            else:
                self.action.append([0.0, 0.0, 0.0, 1.0])

    def build_actor(self):
        with tf.variable_scope("actor/target"):
            b1w = 0.01 * tf.Variable(tf.random_normal([self.N_SLIDING, 256]),name="b1w")
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
            a = tf.nn.softmax(tf.matmul(b5o,b6w)+b6b)
        with tf.variable_scope("actor/eval"):
            b1w = 0.01 * tf.Variable(tf.random_normal([self.N_SLIDING, 256]),name="b1w")
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
            a2 = tf.nn.softmax(tf.matmul(b5o,b6w)+b6b)
        return a,a2

    def read_batch(self):
        s_sliding_list = []
        s_others_list = []
        s_a_list = []
        for i in range(self.BATCH_SIZE):
            index = random.randint(0,len(self.OccState)-1)
            s_sliding_list.append(self.OccState[index])
            s_others_list.append(self.VehState[index])
            s_a_list.append(self.action[index])

        return s_sliding_list,s_others_list,s_a_list

    def run_test(self):

        s_sliding_list, s_others_list, s_a_list = self.read_batch()
        print("a1 = ", self.sess.run(self.a,
                                            {self.tf_s_sliding: s_sliding_list, self.tf_s_others: s_others_list,
                                             self.tf_a: s_a_list}))
        print("a2 = ", self.sess.run(self.a2,
                                            {self.tf_s_sliding: s_sliding_list, self.tf_s_others: s_others_list,
                                             self.tf_a: s_a_list}))

test = Test()
test.read_test_data()
for i in range(10000):
    test.run_test()
