import tensorflow as tf
import numpy as np
import pickle
from math import ceil
from copy import deepcopy
# variable initialization functions
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

class Model:
    def __init__(self, x,y1,y2,y3,y4,y5,keep_prob):
        self.x = x  # input placeholder
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3
        self.y4 = y4
        self.y5 = y5

        self.keep_prob=keep_prob
        # simple 3-layer network 784-64-32-2
        W1 = weight_variable([784,64])
        b1 = bias_variable([64])

        W2 = weight_variable([64,32])
        b2 = bias_variable([32])

        W3 = weight_variable([32, 2])
        b3 = bias_variable([2])

        self.h1 = tf.nn.relu(tf.matmul(x, W1) + b1)  # hidden layer1
        self.h2 = tf.nn.relu(tf.matmul(self.h1, W2) + b2)  # hidden layer1
        self.dropout1 = tf.nn.dropout(self.h2, self.keep_prob)
        self.out1 = tf.matmul(self.dropout1, W3) + b3  # output layer
        self.out2 = tf.matmul(self.dropout1, W3) + b3
        self.out3 = tf.matmul(self.dropout1, W3) + b3
        self.out4 = tf.matmul(self.dropout1, W3) + b3
        self.out5 = tf.matmul(self.dropout1, W3) + b3


        self.probs1 = tf.nn.softmax(self.out1)
        self.probs2 = tf.nn.softmax(self.out2)
        self.probs3 = tf.nn.softmax(self.out3)
        self.probs4 = tf.nn.softmax(self.out4)
        self.probs5 = tf.nn.softmax(self.out5)


        self.var_list = [W1, b1, W2, b2, W3, b3]

        # vanilla single-task loss

        self.cross_entropy1 = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y1, logits=self.out1))
        self.cross_entropy2 = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y2, logits=self.out2))
        self.cross_entropy3 = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y3, logits=self.out3))
        self.cross_entropy4 = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y4, logits=self.out4))
        self.cross_entropy5 = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y5, logits=self.out5))


        self.set_vanilla_loss1()
        self.set_vanilla_loss2()
        self.set_vanilla_loss3()
        self.set_vanilla_loss4()
        self.set_vanilla_loss5()

        # performance metrics
        correct_prediction1 = tf.equal(tf.argmax(self.out1, 1), tf.argmax(self.y1, 1))
        correct_prediction2 = tf.equal(tf.argmax(self.out2, 1), tf.argmax(self.y2, 1))
        correct_prediction3 = tf.equal(tf.argmax(self.out3, 1), tf.argmax(self.y3, 1))
        correct_prediction4 = tf.equal(tf.argmax(self.out4, 1), tf.argmax(self.y4, 1))
        correct_prediction5 = tf.equal(tf.argmax(self.out5, 1), tf.argmax(self.y5, 1))

        self.accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
        self.accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))
        self.accuracy3 = tf.reduce_mean(tf.cast(correct_prediction3, tf.float32))
        self.accuracy4 = tf.reduce_mean(tf.cast(correct_prediction4, tf.float32))
        self.accuracy5 = tf.reduce_mean(tf.cast(correct_prediction5, tf.float32))


        self.omega1 = self.compute_w1()
        self.omega2 = self.compute_w2()
        self.omega3 = self.compute_w3()
        self.omega4 = self.compute_w4()
        self.omega5 = self.compute_w5()

    ###mas global importance:
    def compute_w1(self):
        importance = []
        for v in range(len(self.var_list)):
            im = tf.gradients(tf.nn.l2_loss(self.probs1), self.var_list[v])
            
            importance.append(tf.reduce_mean(tf.abs(im),0))
        return importance

    def compute_w2(self):
        importance = []
        for v in range(len(self.var_list)):
            im = tf.gradients(tf.nn.l2_loss(self.probs2), self.var_list[v])

            importance.append(tf.reduce_mean(tf.abs(im), 0))
        return importance

    def compute_w3(self):
        importance = []
        for v in range(len(self.var_list)):
            im = tf.gradients(tf.nn.l2_loss(self.probs3), self.var_list[v])

            importance.append(tf.reduce_mean(tf.abs(im), 0))
        return importance

    def compute_w4(self):
        importance = []
        for v in range(len(self.var_list)):
            im = tf.gradients(tf.nn.l2_loss(self.probs4), self.var_list[v])

            importance.append(tf.reduce_mean(tf.abs(im), 0))
        return importance

    def compute_w5(self):
        importance = []
        for v in range(len(self.var_list)):
            im = tf.gradients(tf.nn.l2_loss(self.probs5), self.var_list[v])

            importance.append(tf.reduce_mean(tf.abs(im), 0))
        return importance



    def star(self):
        # used for saving optimal weights after most recent task training
        self.star_vars = []

        for v in range(len(self.var_list)):
            self.star_vars.append(self.var_list[v].eval())
        # with open("./optimal_weights.pkl", "wb") as f:
        #     pickle.dump(self.star_vars, f, pickle.HIGHEST_PROTOCOL)

    def restore(self, sess):
        # reassign optimal weights for latest task
        if hasattr(self, "star_vars"):
            for v in range(len(self.var_list)):
                sess.run(self.var_list[v].assign(self.star_vars[v]))

    def set_vanilla_loss1(self):
        self.train_step1 = tf.train.GradientDescentOptimizer(0.001).minimize(self.cross_entropy1)
    def set_vanilla_loss2(self):
        self.train_step2 = tf.train.GradientDescentOptimizer(0.001).minimize(self.cross_entropy2)
    def set_vanilla_loss3(self):
        self.train_step3 = tf.train.GradientDescentOptimizer(0.001).minimize(self.cross_entropy3)
    def set_vanilla_loss4(self):
        self.train_step4 = tf.train.GradientDescentOptimizer(0.001).minimize(self.cross_entropy4)
    def set_vanilla_loss5(self):
        self.train_step5 = tf.train.GradientDescentOptimizer(0.001).minimize(self.cross_entropy5)

    def update_MAS_loss1(self, lam):
        if not hasattr(self, "MAS_loss1"):
            self.MAS_loss1 = self.cross_entropy1
        for v in range(len(self.var_list)):
            self.MAS_loss1 += (lam / 2) * tf.reduce_sum(tf.multiply(self.last_omega[v], tf.square(self.var_list[v] - self.star_vars[v])))
        self.train_step1 = tf.train.GradientDescentOptimizer(0.001).minimize(self.MAS_loss1)
    def update_MAS_loss2(self, lam):
        if not hasattr(self, "MAS_loss2"):
            self.MAS_loss2 = self.cross_entropy2
        for v in range(len(self.var_list)):
            self.MAS_loss2 += (lam / 2) * tf.reduce_sum(tf.multiply(self.last_omega[v], tf.square(self.var_list[v] - self.star_vars[v])))
        self.train_step2 = tf.train.GradientDescentOptimizer(0.001).minimize(self.MAS_loss2)
    def update_MAS_loss3(self, lam):
        if not hasattr(self, "MAS_loss3"):
            self.MAS_loss3 = self.cross_entropy3
        for v in range(len(self.var_list)):
            self.MAS_loss3 += (lam / 2) * tf.reduce_sum(tf.multiply(self.last_omega[v], tf.square(self.var_list[v] - self.star_vars[v])))
        self.train_step3 = tf.train.GradientDescentOptimizer(0.001).minimize(self.MAS_loss3)
    def update_MAS_loss4(self, lam):
        if not hasattr(self, "MAS_loss4"):
            self.MAS_loss4 = self.cross_entropy4
        for v in range(len(self.var_list)):
            self.MAS_loss4 += (lam / 2) * tf.reduce_sum(tf.multiply(self.last_omega[v], tf.square(self.var_list[v] - self.star_vars[v])))
        self.train_step4 = tf.train.GradientDescentOptimizer(0.001).minimize(self.MAS_loss4)
    def update_MAS_loss5(self, lam):
        if not hasattr(self, "MAS_loss5"):
            self.MAS_loss5 = self.cross_entropy5
        for v in range(len(self.var_list)):
            self.MAS_loss5 += (lam / 2) * tf.reduce_sum(tf.multiply(self.last_omega[v], tf.square(self.var_list[v] - self.star_vars[v])))
        self.train_step5 = tf.train.GradientDescentOptimizer(0.001).minimize(self.MAS_loss5)


    def compute_omega1(self, sess, imgset, batch_size,keep_prob):
        Omega_M = []
        batch_num = ceil(len(imgset) / batch_size)
        for iter in range(batch_num):
            index = iter * batch_size
            Omega_M.append(sess.run(self.omega1, feed_dict={self.x: imgset[index:(index + batch_size - 1)],self.keep_prob:keep_prob}))
        self.Omega_M = np.sum(Omega_M, 0) / batch_num
        return self.Omega_M
    def compute_omega2(self, sess, imgset, batch_size,keep_prob):
        Omega_M = []
        batch_num = ceil(len(imgset) / batch_size)
        for iter in range(batch_num):
            index = iter * batch_size
            Omega_M.append(sess.run(self.omega2, feed_dict={self.x: imgset[index:(index + batch_size - 1)],self.keep_prob:keep_prob}))
        self.Omega_M = np.sum(Omega_M, 0) / batch_num
        return self.Omega_M
    def compute_omega3(self, sess, imgset, batch_size,keep_prob):
        Omega_M = []
        batch_num = ceil(len(imgset) / batch_size)
        for iter in range(batch_num):
            index = iter * batch_size
            Omega_M.append(sess.run(self.omega3, feed_dict={self.x: imgset[index:(index + batch_size - 1)],self.keep_prob:keep_prob}))
        self.Omega_M = np.sum(Omega_M, 0) / batch_num
        return self.Omega_M
    def compute_omega4(self, sess, imgset, batch_size,keep_prob):
        Omega_M = []
        batch_num = ceil(len(imgset) / batch_size)
        for iter in range(batch_num):
            index = iter * batch_size
            Omega_M.append(sess.run(self.omega4, feed_dict={self.x: imgset[index:(index + batch_size - 1)],self.keep_prob:keep_prob}))
        self.Omega_M = np.sum(Omega_M, 0) / batch_num
        return self.Omega_M
    def compute_omega5(self, sess, imgset, batch_size,keep_prob):
        Omega_M = []
        batch_num = ceil(len(imgset) / batch_size)
        for iter in range(batch_num):
            index = iter * batch_size
            Omega_M.append(sess.run(self.omega5, feed_dict={self.x: imgset[index:(index + batch_size - 1)],self.keep_prob:keep_prob}))
        self.Omega_M = np.sum(Omega_M, 0) / batch_num
        return self.Omega_M

    def save_omega(self):
        self.last_omega=deepcopy(self.Omega_M)
        

    def omega_accumulate(self):
        self.Omega_M=self.last_omega+self.Omega_M
        return self.Omega_M

