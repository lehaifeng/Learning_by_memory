import tensorflow as tf
import numpy as np
import pickle
from copy import deepcopy

from tensorflow.examples.tutorials.mnist import input_data

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
        y=label_to_one_hot(y,2)
    # Only complete batches are submitted
    for i in range(len(X) // batchsize):
        yield X[i * batchsize:(i + 1) * batchsize], y[i * batchsize:(i + 1) * batchsize]

data01=read_batch("../data/train01")
test01=read_batch("../data/test01")
#data_task2
data23=read_batch("../data/train23")
test23=read_batch("../data/test23")
#data_task3
data45=read_batch("../data/train45")
test45=read_batch("../data/test45")
#data_task4
data67=read_batch("../data/train67")
test67=read_batch("../data/test67")
#data_task5
data89=read_batch("../data/train89")
test89=read_batch("../data/test89")

tasks_train = [data01,data23,data45,data67,data89]
tasks_test = [test01,test23,test45,test67,test89]

# network
num_tasks_to_run = 5

num_epochs_per_task = 50


# Parameters for the intelligence synapses model.
param_c = 0.1
param_xi = 0.1


minibatch_size = 256
learning_rate = 0.01



def weight_variable(input_size, output_size):
	return tf.Variable( tf.random_uniform([input_size,output_size], -1.0/np.sqrt(input_size), 1.0/np.sqrt(input_size)) )



## Network definition -- a simple MLP with 2 hidden layers
x = tf.placeholder(tf.float32, shape=[None, 784])
y_tgt = tf.placeholder(tf.float32, shape=[None, 2])


# Note: the main paper uses a larger network + dropout; both significantly improve the performance of the system.
N1 = 64
N2 = 32

W1 = weight_variable(784,N1)
b1 = tf.Variable(tf.zeros([1,N1]))

W2 = weight_variable(N1,N2)
b2 = tf.Variable(tf.zeros([1,N2]))

Wo = weight_variable(N2,2)
bo = tf.Variable(tf.zeros([1,2]))


h1 = tf.nn.relu( tf.matmul(x,W1) + b1 )
h2 = tf.nn.relu( tf.matmul(h1,W2) + b2 )
y = tf.nn.softmax( tf.matmul(h2,Wo) + bo )


cross_entropy = -tf.reduce_sum( y_tgt*tf.log(y+1e-04) + (1.-y_tgt)*tf.log(1.-y+1e-04) )

optimizer = tf.train.GradientDescentOptimizer(learning_rate)

## Implementation of the intelligent synapses model
variables = [W1, b1, W2, b2, Wo, bo]

small_omega_var = {}
previous_weights_mu_minus_1 = {}
big_omega_var = {}
aux_loss = 0.0

reset_small_omega_ops = []
update_small_omega_ops = []
update_big_omega_ops = []
for var in variables:
	small_omega_var[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
	previous_weights_mu_minus_1[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
	big_omega_var[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)

	aux_loss += tf.reduce_sum(tf.multiply( big_omega_var[var.op.name], tf.square(previous_weights_mu_minus_1[var.op.name] - var) ))

	reset_small_omega_ops.append( tf.assign( previous_weights_mu_minus_1[var.op.name], var ) )
	reset_small_omega_ops.append( tf.assign( small_omega_var[var.op.name], small_omega_var[var.op.name]*0.0 ) )

	update_big_omega_ops.append( tf.assign_add( big_omega_var[var.op.name],  tf.div(small_omega_var[var.op.name],(param_xi + tf.square(var-previous_weights_mu_minus_1[var.op.name]) ))   ) )

# After each task is complete, call update_big_omega and reset_small_omega
update_big_omega = tf.group(*update_big_omega_ops)

# Reset_small_omega also makes a backup of the final weights, used as hook in the auxiliary loss
reset_small_omega = tf.group(*reset_small_omega_ops)


# Gradient of the loss function for the current task
gradients = optimizer.compute_gradients(cross_entropy, var_list=variables)

# Gradient of the loss+aux function, in order to both perform training and to compute delta_weights
gradients_with_aux = optimizer.compute_gradients(cross_entropy + param_c*aux_loss, var_list=variables)

for i, (grad,var) in enumerate(gradients_with_aux):
	update_small_omega_ops.append( tf.assign_add( small_omega_var[var.op.name], learning_rate*gradients_with_aux[i][0]*gradients[i][0] ) ) # small_omega -= delta_weight(t)*gradient(t)

update_small_omega = tf.group(*update_small_omega_ops) # 1) update small_omega after each train!


train = optimizer.apply_gradients(gradients_with_aux)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_tgt,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


## Initialize session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())

avg_performance = []
first_performance = []
last_performance = []
for task in range(num_tasks_to_run):
	# print "Training task: ",task+1,"/",num_tasks_to_run
	print("\t task ", task+1)

	for epoch in range(num_epochs_per_task):
		if epoch%5==0:
			print("\t Epoch ",epoch)
		for images, labels in yield_mb(tasks_train[task]["data"], tasks_train[task]["label"], batchsize=256, shuffle=True, one_hot=True):
			sess.run([train, update_small_omega], feed_dict={x: images, y_: labels})

	sess.run( update_big_omega )
	sess.run( reset_small_omega )

	# Print test set accuracy to each task encountered so far
	avg_accuracy = 0.0
	for test_task in range(task+1):
		# test_images = mnist.test.images
		test_data = tasks_test[test_task]

		acc = sess.run(accuracy, feed_dict={x: test_data["data"], y_: label_to_one_hot(test_data["label"],2)}) * 100.0
		avg_accuracy += acc

		if test_task == 0:
			first_performance.append(acc)
		if test_task == task:
			last_performance.append(acc)

		print("Task: ",test_task," \tAccuracy: ",acc)

	avg_accuracy = avg_accuracy/(task+1)
	print("Avg Perf: ",avg_accuracy)

	avg_performance.append( avg_accuracy )

print("avg_accuracy: ", avg_accuracy)
print("avg_performance: ", avg_performance)
print("first_performance: ", first_performance)
print("last_performance: ", last_performance)


