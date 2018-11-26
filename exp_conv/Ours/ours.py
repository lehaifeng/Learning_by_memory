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
		new_image = (old_image - np.mean(old_image)) / np.std(old_image)
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
    x_train =  (x_train-np.mean(x_train)) / np.std(x_train, axis=0)
    x_test = (x_test-np.mean(x_test	)) / np.std(x_test, axis=0)
    # Reshape
    x_train = x_train.reshape(-1, 32, 32, 3)
    x_test = x_test.reshape(-1, 32, 32, 3)
	
    #print(x_train.shape[0])
    #x_train = data_augmentation(x_train)
    #print(x_train.shape[0])

    #y_train = np.concatenate((y_train, y_train, y_train), axis=0)
    #print(y_train.shape[0])
	
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
	
    print("cifar10_train:", x_train.shape, y_train.shape)
    print("cifar10_test:", x_train.shape, y_train.shape)
    #dateprocess
	#data augmentation

    return x_train, x_test, y_train, y_test

def stl_for_library(): 
    # Raw data
    x_train, x_test, y_train, y_test = get_stl()
    x_train = x_train.reshape(-1, 96, 96, 3)
    x_test = x_test.reshape(-1, 96, 96, 3)

    x_train = data_augmentation(x_train)
    
    x_train =  (x_train-np.mean(x_train)) / np.std(x_train, axis=0)
    x_test = (x_test-np.mean(x_test	)) / np.std(x_test, axis=0)
    shape = [32, 32, 3]
    x_test = image_crop(x_test, shape)
    x_train = image_crop(x_train, shape)
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

    print("stl10_train:", x_train.shape, y_train.shape)
    print("stl10_test:", x_train.shape, y_train.shape)
    return x_train, x_test, y_train, y_test

def svhn_for_library(): 
    # Raw data
    x_train, y_train, x_test, y_test  = get_svhn()

    x_train = x_train.reshape(-1, 32, 32, 3)
    x_test = x_test.reshape(-1, 32, 32, 3)
    
    x_train =  (x_train-np.mean(x_train)) / np.std(x_train, axis=0)
    x_test = (x_test-np.mean(x_test	)) / np.std(x_test, axis=0)
    shape = [32, 32, 3]
    x_test = image_crop(x_test, shape)
    #x_train = data_augmentation(x_train)
    #print(x_train.shape)
    #y_train = np.concatenate((y_train, y_train, y_train), axis=0)
    #print(y_train.shape)
    assert len(x_train) == len(y_train)
    #print(y_train.shape[0])
    
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    #dateprocess
	#data augmentation

    print("svhn_train:", x_train.shape, y_train.shape)
    print("svhn_test:", x_test.shape, y_test.shape)
    return x_train, x_test, y_train, y_test

def notMNIST_for_library():
    path = "../data/notMNIST/notMNIST.pickle"
    x_train, y_train, x_test, y_test = get_notmnist(path)

    x_train = x_train.reshape(-1, 28,28, 3)
    x_test = x_test.reshape(-1, 28, 28, 3)
    
    x_train =  (x_train-np.mean(x_train)) / np.std(x_train, axis=0)
    x_test = (x_test-np.mean(x_test	)) / np.std(x_test, axis=0)
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
	
    print("notMNIST_train:", x_train.shape, y_train.shape)
	
    print("notMNIST_test:", x_test.shape, y_test.shape)
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
    x_train =  (x_train-np.mean(x_train)) / np.std(x_train, axis=0)
    x_test = (x_test-np.mean(x_test	)) / np.std(x_test, axis=0)
    #print(x_train.shape)
    #print(y_train.shape)
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
	
    print("MNIST_train:", x_train.shape, y_train.shape)
    print("MNIST_test:", x_test.shape, y_test.shape)
    return x_train, x_test, y_train, y_test
def shuffle_data(X, y):
    s = np.arange(len(X))
    np.random.shuffle(s)
    X = X[s]
    y = y[s]
    return X, y


def yield_mb(X, y, batchsize=256, shuffle=False, one_hot=False):
    #assert len(X) == len(y)
    if shuffle:
        X, y = shuffle_data(X, y)
    if one_hot:
        y=label_to_one_hot(y, 10)
    # Only complete batches are submitted
    for i in range(len(X) // batchsize):
        yield X[i * batchsize:(i + 1) * batchsize], y[i * batchsize:(i + 1) * batchsize]


def compute_omega(sess, imgset, batch_size, param_im):
	Omega_M = []
	batch_num = ceil(len(imgset) / batch_size)
	for iter in range(batch_num):
		index = iter * batch_size
		Omega_M.append(sess.run(param_im, feed_dict={x: imgset[index:(index + batch_size - 1)], keep_prob:1}))
	Omega_M = np.sum(Omega_M, 0) / batch_num

	return Omega_M


def optimizer(lr_start,cost,varlist,steps,epoches):
	learning_rate = tf.train.exponential_decay(lr_start, global_step=steps, decay_steps=epoches, decay_rate=0.9, staircase=True) # lr decay=1
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,global_step=steps, var_list=varlist)
	return train_step,learning_rate

# calculate params importance
def Cal_importance(varlist,output):
	importance = []
	probs = tf.nn.softmax(output)
	for v in range(len(varlist)):
		# # information entropy
		I = -tf.reduce_sum(probs * tf.log(probs))
		gradients = tf.gradients(I, varlist[v])
		importance.append(tf.reduce_mean(
			tf.maximum((gradients * varlist[v] + 1 / 2 * tf.multiply(tf.square(varlist[v]), tf.square(gradients))), 0),0))
	return importance


def im_sharpen(im_value,sharpen=True,beta=1):
	O_im = []
	if sharpen:
		total = 0
		for j in im_value:
			total += np.sum(np.exp(beta * j))
		for i in im_value:
			temp = np.exp(beta * i) / total
			O_im.append(temp)
	else:
		O_im = im_value
	return O_im

# update loss function, with auxiliary loss
def update_loss(pre_var,Omega,var,new_loss,lamb):
	aux_loss = 0
	for v in range(len(pre_var)):
		aux_loss += tf.reduce_sum(tf.multiply(Omega[v], tf.square(pre_var[v] - var[v])))
	loss = new_loss + lamb * aux_loss
	return loss


def clone_net(net_old, net):
	for i in range(len(net.params)):
		assign_op = net_old.params[i].assign(net.params[i])
		sess.run(assign_op)


x0, tx0, y0, ty0 = notMNIST_for_library()
x1, tx1, y1, ty1 = MNIST_for_library()
x2, tx2, y2, ty2 = svhn_for_library()
x3, tx3, y3, ty3 = stl_for_library()
x4, tx4, y4, ty4 = cifar_for_library()

tasks_train =  [x0, x1, x2, x3, x4]
tasks_test = [tx0, tx1, tx2, tx3, tx4]
tasks_train_label = [y0, y1, y2, y3, y4]
tasks_test_label = [ty0, ty1, ty2, ty3, ty4]
# train and test setting
num_tasks_to_run = 5
num_epochs_per_task = 50
minibatch_size = 512
Lamb_set = [0.01,0.1,0.5,1,2,4,6,8,12] # hyper-parameter1 loss
#Lamb_set = [0.01]
epoches = 20*50000/minibatch_size #decay per 10 epoch
Beta = 1 # hyper-parameter2 sharpen 
learning_r = 0.01
keep_prob = tf.placeholder(tf.float32)

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
model = Model(x, keep_prob)
model_old = Model(x, keep_prob)

y_ = tf.placeholder(tf.float32, shape=[None, 10])
out_dim = int(y_.get_shape()[1])  # 10 for MNIST
lr_steps = tf.Variable(0)

# expand output layer
W_output = []
b_output = []
task_output = []
for task in range(num_tasks_to_run):
	W = weight_variable([512, out_dim])
	b = bias_variable([out_dim])
	W_output.append(W)
	b_output.append(b)
for task in range(num_tasks_to_run):
	output = tf.matmul(model.fc1_drop, W_output[task]) + b_output[task]
	task_output.append(output)


config = tf.ConfigProto(allow_soft_placement=True) # config
config.gpu_options.allow_growth = True


per_avg_performance = []
best_accuracy = 0.0
list_acc = []
for Lamb in Lamb_set:
	sess = tf.InteractiveSession(config=config)
#	writer = tf.summary.FileWriter("logs/", sess.graph)
	sess.run(tf.global_variables_initializer())

	Omega_v = []
	first_performance = []
	last_performance = []
	tasks_acc = []

	for task in range(num_tasks_to_run):
		# print "Training task: ",task+1,"/",num_tasks_to_run
		print("\t task ", task + 1)

		if task == 0:
			num_epochs_per_task = 15
			learning_r = 0.01
			cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=task_output[task]))
		else:
			if task == 1 :
				num_epochs_per_task = 50
				learning_r = 0.001
			if task == 2:
				num_epochs_per_task = 50
			if task == 3:
				num_epochs_per_task = 50
				learning_r = 0.005
			clone_net(model_old, model)
			cost_new = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=task_output[task]))
			cost = update_loss(pre_var=model_old.params, Omega=Omega_v, var=model.params, new_loss=cost_new, lamb=Lamb)
		train_op,learning_rate = optimizer(lr_start=learning_r, cost=cost, varlist=[model.params, W_output[task], b_output[task]],steps=lr_steps,epoches=epoches)
		sess.run(tf.variables_initializer([lr_steps]))

		correct_prediction = tf.equal(tf.argmax(task_output[task], 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		
		for epoch in range(num_epochs_per_task):
			if epoch % 5 == 0:
				print("\t Epoch ", epoch)
				acc = 0
				step = 0
				for images, labels in yield_mb(tasks_test[task], tasks_test_label[task], batchsize=128,
					shuffle=False,one_hot=True):
					acc += sess.run(accuracy, feed_dict={x: images, y_: labels, keep_prob:1})
					step += 1
				acc = acc / step
				print("Task: ", task, " \tAccuracy: ", acc)	
				cost_ = cost.eval(feed_dict={x:images, y_:labels, keep_prob:1})
				print("loss: ", cost_)
				lea = learning_rate.eval()
				print("lea: ", lea)
			for images, labels in yield_mb( tasks_train[task], tasks_train_label[task], batchsize=64, shuffle=True, one_hot=True):
				sess.run(train_op, feed_dict={x: images, y_: labels, keep_prob:0.5})
		# calculate params importance
		param_importance = Cal_importance(varlist=model.params, output=task_output[task])

		# param_importance = Cal_importance(varlist=model.params, output=task_output[task],y_tgt=y_) # loss
		if task == 0:
			Omega_v = compute_omega(sess, tasks_train[task], batch_size=100,param_im=param_importance)

		else:
			for l in range(len(Omega_v)):
				Omega_v[l] = Omega_v[l] + compute_omega(sess,  tasks_train[task], batch_size=100, param_im=param_importance)[l]

			# sharpen
			Omega_v = im_sharpen(Omega_v, sharpen=False, beta=Beta)
		# Print test set accuracy to each task encountered so far
		avg_accuracy = 0.0
		for test_task in range(task + 1):
			#print("fuck test")
			
			correct_prediction = tf.equal(tf.argmax(task_output[test_task], 1), tf.argmax(y_, 1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

			acc = []

			for images, labels in yield_mb(tasks_test[task], tasks_test_label[task], batchsize=256, shuffle=True,
											one_hot=True):
				acc.append(sess.run(accuracy,feed_dict={x: images, y_: labels, keep_prob: 1}))

			avg_ = np.mean(acc)
			avg_accuracy += avg_
			tasks_acc.append(avg_)
			print("Task: ", test_task, " \tAccuracy: ", avg_)

			avg_accuracy = avg_accuracy / (task + 1)
			if test_task == 0:
				first_performance.append(acc)
			if test_task == task:
				last_performance.append(acc)
			tasks_acc.append(acc)
			print("Task: ", test_task, " \tAccuracy: ", acc)

		avg_accuracy = avg_accuracy / (task + 1)
		print("Avg Perf: ", avg_accuracy)


	if avg_accuracy > best_accuracy:
		best_accuracy = avg_accuracy
		best_acc = tasks_acc
	print('best-aver-acc ',best_accuracy)
	per_avg_performance.append(avg_accuracy)
	list_acc.append(tasks_acc)

best_lamb = Lamb_set[per_avg_performance.index(max(per_avg_performance))]
print('best lamb is: ',best_lamb)
#print(list_acc)
# save best model-test log
file= open('log-ours.txt', 'w')
for fp in best_acc:
	file.write(str(fp))
	file.write('\n')
file.write(str(per_avg_performance))
file.close()


# Acc
ACC = best_accuracy

sess.close()
