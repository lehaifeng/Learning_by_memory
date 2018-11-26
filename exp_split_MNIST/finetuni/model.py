import tensorflow as tf


def weight_variable(shape):
	with tf.name_scope('weights'):
		initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	with tf.name_scope('biases'):
		initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

class Model:
	def __init__(self, x):
		# create network 784-50-50-10
		in_dim = int(x.get_shape()[1]) # 784 for MNIST

		self.x = x # input placeholder

		keep_prob = 0.5
		# simple 3-layer network
		self.W1 = weight_variable([in_dim,64])
		self.b1 = bias_variable([64])
		self.W2 = weight_variable([64, 32])
		self.b2 = bias_variable([32])

		self.h1 = tf.nn.relu(tf.matmul(self.x, self.W1) + self.b1)  # hidden layer
		self.h1_drop = tf.nn.dropout(self.h1, keep_prob)

		self.h2 = tf.nn.relu(tf.matmul(self.h1_drop,self.W2) + self.b2) # hidden layer
		self.h2_drop = tf.nn.dropout(self.h2,keep_prob)
		self.params = [self.W1,self.b1,self.W2,self.b2]
		return