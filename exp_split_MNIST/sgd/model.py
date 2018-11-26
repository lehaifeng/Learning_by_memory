import tensorflow as tf
import numpy as np
import pickle
from math import ceil

# variable initialization functions
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

class Model:
    def __init__(self, x, y_,keep_prob):

        self.x = x  # input placeholder
        self.y_ = y_
        self.keep_prob=keep_prob
        # simple 3-layer network 784-64-32-10
        W1 = weight_variable([784,64])
        b1 = bias_variable([64])

        W2 = weight_variable([64,32])
        b2 = bias_variable([32])

        W3 = weight_variable([32, 2])
        b3 = bias_variable([2])

        self.h1 = tf.nn.relu(tf.matmul(x, W1) + b1)  # hidden layer1
        self.h2 = tf.nn.relu(tf.matmul(self.h1, W2) + b2)  # hidden layer1
        self.dropout1=tf.nn.dropout(self.h2,self.keep_prob)
        self.y = tf.matmul(self.dropout1, W3) + b3  # output layer

        self.probs = tf.nn.softmax(self.y)

        self.var_list = [W1, b1, W2, b2, W3, b3]


        # vanilla single-task loss
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))
        self.set_vanilla_loss()

        # performance metrics
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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

    def set_vanilla_loss(self):
        self.train_step = tf.train.GradientDescentOptimizer(0.001).minimize(self.cross_entropy)