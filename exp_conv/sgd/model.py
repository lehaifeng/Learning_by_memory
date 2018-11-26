import tensorflow as tf
import numpy as np
import pickle
from math import ceil

def weight_variable(shape):
	with tf.name_scope('weights'):
		initial = tf.contrib.layers.xavier_initializer()
	return tf.Variable(initial(shape))

def bias_variable(shape):
	with tf.name_scope('biases'):
		initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


class Model:
    def __init__(self, x, y_,keep_prob):

        self.x = x  # input placeholder
        self.y_ = y_
        self.keep_prob=keep_prob

        self.W1 = weight_variable([3, 3, 3, 64])
        self.b1 = bias_variable([64])
        self.h_conv1 = tf.nn.relu(conv2d(x, self.W1) + self.b1)
        self.h_pool1 = max_pool2x2(self.h_conv1)

        self.W2 = weight_variable([3, 3, 64, 128])
        self.b2 = bias_variable([128])
        self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W2) + self.b2)
        self.h_pool2 = max_pool2x2(self.h_conv2)

        self.W3 = weight_variable([3, 3, 128, 256])
        self.b3 = bias_variable([256])
        self.h_conv3 = tf.nn.relu(conv2d(self.h_pool2, self.W3) + self.b3)
        self.h_pool3 = max_pool2x2(self.h_conv3)

        self.W4 = weight_variable([4*4*256, 512])
        self.b4 = bias_variable([512])
        self.h_pool2_flat = tf.reshape(self.h_pool3, [-1, 4*4*256])

        self.fc1 = tf.nn.relu( tf.matmul(self.h_pool2_flat, self.W4) + self.b4)
        self.fc1_drop = tf.nn.dropout(self.fc1, keep_prob)

        self.W5 = weight_variable([512, 10])
        self.b5 = bias_variable([10])

        self.y = tf.matmul(self.fc1_drop, self.W5) + self.b5  # output layer

        self.probs = tf.nn.softmax(self.y)

        self.var_list = [self.W1,self.b1,self.W2,self.b2,self.W3,self.b3,self.W4,self.b4,self.W5,self.b5]


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
        self.train_step = tf.train.GradientDescentOptimizer(0.01).minimize(self.cross_entropy)