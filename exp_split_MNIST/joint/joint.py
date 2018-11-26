import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pickle
import numpy as np
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

class Model:
    def __init__(self, x, y_, keep_prob):
        self.x = x  # input placeholder
        self.y_ = y_
        self.keep_prob=keep_prob
        W1 = weight_variable([784, 512])
        b1 = bias_variable([512])

        W2 = weight_variable([512, 256])
        b2 = bias_variable([256])

        W3 = weight_variable([256, 10])
        b3 = bias_variable([10])
        
        self.h1 = tf.nn.relu(tf.matmul(x, W1) + b1)  # hidden layer1
        self.h2 = tf.nn.relu(tf.matmul(self.h1, W2) + b2)  # hidden layer1
        self.dropout1=tf.nn.dropout(self.h2,self.keep_prob)
        self.y = tf.matmul(self.dropout1, W3) + b3  # output layer

        self.probs = tf.nn.softmax(self.y)

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))


        # performance metrics
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.train()

    def train(self):
        self.train_step = tf.train.GradientDescentOptimizer(0.001).minimize(self.cross_entropy)


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
def read_batch(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def shuffle_data(X, y):
    s = np.arange(len(X))
    np.random.shuffle(s)
    X = X[s]
    y = y[s]
    return X, y

def label_to_one_hot(y,C):
    return np.eye(C)[y.reshape(-1)]

def yield_mb(X, y, batchsize=256, shuffle=False,one_hot=False):
    assert len(X) == len(y)
    if shuffle:
        X, y = shuffle_data(X, y)
    if one_hot:
        y=label_to_one_hot(y,10)
    # Only complete batches are submitted
    for i in range(len(X) // batchsize):
        yield X[i * batchsize:(i + 1) * batchsize], y[i * batchsize:(i + 1) * batchsize]

test01=read_batch("../data/test01")
#data_task2
test23=read_batch("../data/test23")
#data_task3
test45=read_batch("../data/test45")
#data_task4
test67=read_batch("../data/test67")
#data_task5
test89=read_batch("../data/test89")

tasks_test = [test01,test23,test45,test67,test89]

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)
model = Model(x, y_, keep_prob)

train_epoches = 1000
batch_size = 128

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
display_step = 100

for epoch in range(train_epoches):
    loss = 0.
    total_batch = int(mnist.train.num_examples/batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size) 
        sess.run(model.train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
    # Display logs per epoch step
    if epoch % display_step == 0:
        test_acc_2 = sess.run(model.accuracy, feed_dict={x:test23["data"], y_:label_to_one_hot(test23["label"],10),keep_prob:1})
        print("the 2th test acc: ", test_acc_2)

        test_acc = sess.run(model.accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1})
        print("the average test:", test_acc)
print("DONE")