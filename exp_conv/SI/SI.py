import numpy as np
from math import ceil
from model import *
import pickle
import os
import cv2
import random
import sys
import gc
import struct
from copy import deepcopy
import gc
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.io import loadmat as load

def read_batch(src):
    '''Unpack the pickle files
    '''
    with open(src, 'rb') as f:
        data = pickle.load(f,encoding='latin1')
    return data


def get_MNIST(path, kind="train"): 
	train_labels = os.path.join(path, '%s-labels.idx1-ubyte' % kind) 
	train_data = os.path.join(path, '%s-images.idx3-ubyte' % kind) 
	with open(train_labels, 'rb') as lbpath: 
		magic, n = struct.unpack('>II', lbpath.read(8))
		labels = np.fromfile(lbpath, dtype=np.uint8) 
	with open(train_data, 'rb') as imgpath: 
		magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16)) 
		images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784) 
	return images, labels 

def get_notmnist(pickle_file):

	with open(pickle_file, 'rb') as f:
		save = pickle.load(f)
		x_train = save['train_dataset']
		y_train = save['train_labels']
		x_test = save['test_dataset']
		y_test = save['test_labels']
		del save  # hint to help gc free up memory
		gc.collect()
		#print('Training set', x_train.shape, y_train.shape)
		#print('Test set', x_test.shape, y_test.shape)
	return x_train, y_train, x_test, y_test

def reformat(samples, labels):
    samples = np.transpose(samples, (3, 0, 1, 2))
    labels = np.array([x[0] for x in labels])
    lbs = []
    for num in labels:
        tmp_lbs = num - 1
        lbs.append(tmp_lbs)
    #print(labels[:100])
    #print(lbs[:100])
    return samples, lbs
def get_svhn():
    
    traindata = load('../data/SVHN/svhn/train_32x32.mat')
    testdata = load('../data/SVHN/svhn/test_32x32.mat')
    
    train_samples = traindata['X']
    train_labels = traindata['y']
    test_samples = testdata['X']
    test_labels = testdata['y']
    x_train, y_train = reformat(train_samples, train_labels)
    x_test, y_test = reformat(test_samples, test_labels)
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)



def get_stl():
	"""A generator which yields a 4-D numpy tensor of size batch,height,width,depth and a size of batch,labels label.."""
	y_train = None
	y_test = None
	x_train = None
	x_test = None
	with open("../data/STL/STL-10/train_y.bin", "rb") as fin:
		y_train = np.fromfile(fin, dtype=np.uint8)
	with open("../data/STL/STL-10/train_X.bin", "rb") as fin:
		x_train = np.fromfile(fin, dtype=np.uint8)
		
		x_train = np.asarray(x_train / 255.0, dtype=np.float) # TODO: Divide by 255?
	with open("../data/STL/STL-10/test_y.bin", "rb") as fin:
		y_test = np.fromfile(fin, dtype=np.uint8)
	with open("../data/STL/STL-10/test_X.bin", "rb") as fin:
		x_test = np.fromfile(fin, dtype=np.uint8)
		x_test = np.asarray(x_test / 255.0, dtype=np.float) # TODO: Divide by 255?
	return x_train, x_test, y_train, y_test

def get_cifar():
    print ('Preparing train set...')
    train_list = [read_batch('../data/cifar10/data_batch_%d'%i) for i in range(1,6)]
    x_train = np.concatenate([t['data'] for t in train_list])
    y_train = np.concatenate([t['labels'] for t in train_list])       
    print ('Preparing test set...')
    tst = read_batch('../data/cifar10/test_batch')
    x_test = tst['data']
    y_test = np.asarray(tst['labels'])
    print ('Done.')

    return x_train, x_test, y_train, y_test

def label_to_one_hot(y,C):
    return np.eye(C)[y.reshape(-1)]


def image_crop(images, shape):
	new_images = []
	for i in range(images.shape[0]):
		old_images = images[i, :, :, :]
		old_images = np.resize(old_images, shape)
		#left = np.random.randint(old_images.shape[0] - shape[0] + 1)
		#top = np.random.randint(old_images.shape[1] - shape[1] + 1)
		#new_image = old_images[left: left+shape[0], top:top+shape[1], :]
		new_images.append(old_images)
	return np.array(new_images)

def image_flip(images):
	for i in range(images.shape[0]):
		old_image = images[i, :, :, :]
		if np.random.random() > 0.5:
			new_image = cv2.flip(old_image, 1)
		else:
			new_image = old_image
		images[i, :, :, :] = new_image
	return images

def image_whitening(images):
	for i in range(images.shape[0]):
		old_image = images[i, :, :, :]
		new_image = (old_image - np.mean(old_image)) / (np.std(old_image)+1e-5)
		images[i, :, :, :] = new_image
	return images

def image_noise(images, mean=0, std=0.01):
    for i in range(images.shape[0]):
        noise = np.random.normal(0,0.1,size=[32, 32, 3])
        old_image = images[i, :, :, :]
        new_image = old_image + noise
        #print(type(new_image))
        images[i] = new_image

    return images
def data_augmentation(images):
	shape = [32, 32, 3]
	#print('crop')
	images = image_crop(images, shape)
	tmp_images = deepcopy(images)
	#print('flip')
	#f_images = image_flip(tmp_images)
	#images = np.append(images, f_images, axis=0)
	#print('whitening')
	w_images = image_whitening(tmp_images)
	images = np.append(images, w_images, axis=0)
	i_images = image_noise(tmp_images)
	images = np.append(images, i_images, axis=0)
	#print('noise')
	return images


def cifar_for_library(): 
    # Raw data
    x_train, x_test, y_train, y_test = get_cifar()
    # Scale pixel intensity
    x_train =  x_train/255.0
    x_test = x_test/255.0
    # Reshape
    x_train = x_train.reshape(-1, 32, 32, 3)
    x_test = x_test.reshape(-1, 32, 32, 3)
    print(x_train.shape[0])
    x_train = data_augmentation(x_train)
    print(x_train.shape[0])

    y_train = np.concatenate((y_train, y_train, y_train), axis=0)
    print(y_train.shape[0])
	
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    #dateprocess
	#data augmentation
    return x_train, x_test, y_train, y_test

def stl_for_library(): 
    # Raw data
    x_train, x_test, y_train, y_test = get_stl()
    x_train = x_train.reshape(-1, 96, 96, 3)
    x_test = x_test.reshape(-1, 96, 96, 3)

    x_train = data_augmentation(x_train)

    shape = [32, 32, 3]
    x_test = image_crop(x_test, shape)

    y_train = np.concatenate((y_train, y_train, y_train), axis=0)
    assert len(x_train) == len(y_train)

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
	
    for i, k in enumerate(y_train):
        y_train[i] = y_train[i] - 1
    for j, m in enumerate(y_test):
        y_test[j] = y_test[j] - 1
    #print(y_train[:3])
    #dateprocess
	#data augmentation

    return x_train, x_test, y_train, y_test

def svhn_for_library(): 
    # Raw data
    x_train, y_train, x_test, y_test  = get_svhn()

    x_train = x_train.reshape(-1, 32, 32, 3)
    x_test = x_test.reshape(-1, 32, 32, 3)

    shape = [32, 32, 3]
    x_test = image_crop(x_test, shape)
    x_train = data_augmentation(x_train)
    print(x_train.shape)
    y_train = np.concatenate((y_train, y_train, y_train), axis=0)
    print(y_train.shape)
    assert len(x_train) == len(y_train)
    #print(y_train.shape[0])
    
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    #dateprocess
	#data augmentation

    return x_train, x_test, y_train, y_test

def notMNIST_for_library():
    path = "../data/notMNIST/notMNIST.pickle"
    x_train, y_train, x_test, y_test = get_notmnist(path)

    x_train = x_train.reshape(-1, 28,28, 3)
    x_test = x_test.reshape(-1, 28, 28, 3)

    shape = [32, 32, 3]
    x_train = image_crop(x_train, shape)
    x_test = image_crop(x_test, shape)
    #x_train = data_augmentation(x_train)
    #y_train = np.concatenate((y_train, y_train, y_train), axis=0)
    assert len(x_train) == len(y_train)
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    return x_train, x_test, y_train, y_test

def add_channel(images):
	new_img = []
	for img in images:
		#print(img.shape)
		img = img.reshape([28, 28])
		#print(img.shape)
		image_data = np.expand_dims(img, axis=2)
		image_data = np.concatenate((image_data, image_data, image_data), axis=-1)
		new_img.append(image_data)
	new_img = np.asarray(new_img)
	#print(new_img.shape)
	return new_img
	
def MNIST_for_library():
    path = "../data/MNIST_data/"
   
    x_train, y_train = get_MNIST(path, kind="train") 
    x_test, y_test = get_MNIST(path, kind="t10k")
    x_train = add_channel(x_train)
    x_test = add_channel(x_test)
    # Scale pixel intensity
    x_train =  x_train/255.0
    x_test = x_test/255.0
    print(x_train.shape)
    print(y_train.shape)
    # Reshape
    #print(type(x_train))
    x_train = x_train.reshape(-1, 28,28, 3)
    x_test = x_test.reshape(-1, 28, 28, 3)

    shape = [32, 32, 3]
    x_train = image_crop(x_train, shape)
    x_test = image_crop(x_test, shape)
    #x_train_da = data_augmentation(x_train)
    #print(x_train.shape[0])
    #x_train = np.append(x_train, x_train_da, axis=0)
    #print(x_train.shape[0])
    #y_train = np.append(y_train, y_train)
    #print(y_train.shape[0])
    
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    return x_train, x_test, y_train, y_test

def shuffle_data(X, y):
    s = np.arange(len(X))
    np.random.shuffle(s)
    X = X[s]
    y = y[s]
    return X, y


def yield_mb(X, y, batchsize=256, shuffle=False, one_hot=False):
    assert len(X) == len(y)
    if shuffle:
        X, y = shuffle_data(X, y)
    if one_hot:
        y=label_to_one_hot(y, 10)
    # Only complete batches are submitted
    for i in range(len(X) // batchsize):
        yield X[i * batchsize:(i + 1) * batchsize], y[i * batchsize:(i + 1) * batchsize]


num_tasks_to_run = 5

num_epochs_per_task = 50


# Parameters for the intelligence synapses model.
param_c = 0.1
param_xi = 0.1


minibatch_size = 256
learning_rate = 0.01



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


## Network definition -- a simple MLP with 2 hidden layers
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_tgt = tf.placeholder(tf.float32, shape=[None, 10])





W1 = weight_variable([3, 3, 3, 64])
b1 = bias_variable([64])
h_conv1 = tf.nn.relu(conv2d(x, W1) + b1)
h_pool1 = max_pool2x2(h_conv1)
W2 = weight_variable([3, 3, 64, 128])
b2 = bias_variable([128])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W2) + b2)
h_pool2 = max_pool2x2(h_conv2)

W3 = weight_variable([3, 3, 128, 256])
b3 = bias_variable([256])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W3) + b3)
h_pool3 = max_pool2x2(h_conv3)

W4 = weight_variable([4*4*256, 512])
b4 = bias_variable([512])
h_pool2_flat = tf.reshape(h_pool3, [-1, 4*4*256])
		
W5 = weight_variable([512, 10])
b5 = bias_variable([10])

fc1 = tf.nn.relu( tf.matmul(h_pool2_flat, W4) + b4)
fc1_drop = tf.nn.dropout(fc1, 0.5)


y = tf.nn.softmax( tf.matmul(fc1_drop,W5) + b5 )
cross_entropy = -tf.reduce_sum( y_tgt*tf.log(y+1e-04) + (1.-y_tgt)*tf.log(1.-y+1e-04) )

optimizer = tf.train.GradientDescentOptimizer(learning_rate)

## Implementation of the intelligent synapses model
variables = [W1, b1, W2, b2, W3, b3, W4, b4, W5, b5]

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


# load data
x0, tx0, y0, ty0 = cifar_for_library()
x1, tx1, y1, ty1 = stl_for_library()
x2, tx2, y2, ty2 = svhn_for_library()
x3, tx3, y3, ty3 = notMNIST_for_library()
x4, tx4, y4, ty4 = MNIST_for_library()

tasks_train = [x0, x1, x2, x3, x4]
tasks_test = [tx0, tx1, tx2, tx3, tx4]

tasks_train_label = [y0, y1, y2, y3, y4]
tasks_test_label = [ty0, ty1, ty2, ty3, ty4]


avg_performance = []
first_performance = []
last_performance = []
for task in range(num_tasks_to_run):
	# print "Training task: ",task+1,"/",num_tasks_to_run
	print("\t task ", task+1)

	for epoch in range(num_epochs_per_task):
		if epoch%5==0:
			print("\t Epoch ",epoch)
		for images, labels in yield_mb(tasks_train[task], tasks_train_label[task], batchsize=256, shuffle=True, one_hot=True):
			sess.run([train, update_small_omega], feed_dict={x: images, y_tgt: labels})

	sess.run( update_big_omega )
	sess.run( reset_small_omega )

	# Print test set accuracy to each task encountered so far
	avg_accuracy = 0.0
	for test_task in range(task+1):
		# test_images = mnist.test.images
		test_data = tasks_test[test_task]

		acc = sess.run(accuracy, feed_dict={x: test_data, y_tgt: label_to_one_hot(tasks_test_label[test_task],10)}) * 100.0
		avg_accuracy += acc

		if test_task == 0:
			first_performance.append(acc)
		if test_task == task:
			last_performance.append(acc)

		print("Task: ",test_task," \tAccuracy: ",acc)

	avg_accuracy = avg_accuracy/(task+1)
	print("Avg Perf: ",avg_accuracy)

	avg_performance.append( avg_accuracy )
	# print
	# print
print("avg_accuracy: ", avg_accuracy)
print("avg_performance: ", avg_performance)
print("first_performance: ", first_performance)
print("last_performance: ", last_performance)


