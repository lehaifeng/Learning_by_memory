import tensorflow as tf
import numpy as np
from copy import deepcopy
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import pickle
from model import *
import struct
import tensorflow as tf
from scipy.io import loadmat as load
import os
import gc
# load data

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
    # 图像噪声
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
	
   # print(x_train.shape[0])
    #x_train = data_augmentation(x_train)
    #print(x_train.shape[0])

    #y_train = np.concatenate((y_train, y_train, y_train), axis=0)
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

    #x_train = data_augmentation(x_train)

    shape = [32, 32, 3]
    x_test = image_crop(x_test, shape)

    #y_train = np.concatenate((y_train, y_train, y_train), axis=0)
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
    #x_train = data_augmentation(x_train)
   # print(x_train.shape)
    #y_train = np.concatenate((y_train, y_train, y_train), axis=0)
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
	print(new_img.shape)
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

def optimizer(lr_start,cost,varlist,steps,epoches):
	learning_rate = tf.train.exponential_decay(lr_start, global_step=steps, decay_steps=epoches, decay_rate=0.9,staircase=True) # lr decay=1
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,global_step=steps, var_list=varlist)
	return train_step,learning_rate

def lwf_criterion(t,targets_old,outputs,targets,lamb=10,T=2):
	# Knowledge distillation loss for all previous tasks
	loss_dist=0
	for t_old in range(0,t):
		loss_dist += cross_entropy(outputs[t_old],targets_old[t_old],exp=1/T)

	# Cross entropy loss
	loss_ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=outputs[t]))
	loss = loss_ce+lamb*loss_dist

	return loss,loss_dist,loss_ce

def cross_entropy(outputs, targets, exp, eps=1e-5):
	out = tf.nn.softmax(outputs)
	tar = tf.nn.softmax(targets)
	out = tf.pow(out, exp)
	out = out / tf.expand_dims(tf.reduce_sum(out,axis=1),-1)
	tar = tf.pow(tar, exp)
	tar = tar / tf.expand_dims(tf.reduce_sum(tar, axis=1),-1)
	out = out + eps
	ce = -tf.reduce_mean(tf.reduce_sum(tar * tf.log(out), 1))

	return ce

def clone_net(net_old,net):
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
num_epochs_per_task = 100
minibatch_size = 512
Lamb_set = [0.01,0.1,0.5,1,2,4,6,8,12] # hyper-parameter1 loss
#Lamb_set = [0.01]
epoches = 500*50000/minibatch_size #decay per 10 epoch
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
for task in range(num_tasks_to_run):
	W = weight_variable([32,out_dim])
	b = bias_variable([out_dim])
	W_output.append(W)
	b_output.append(b)
for task in range(num_tasks_to_run):
	output = tf.matmul(model.fc1_drop, W_output[task]) + b_output[task]
	task_output.append(output)

avg_performance = []

## Initialize session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

per_avg_performance = []
best_accuracy = 0.0

for Lamb in Lamb_set:
	# create and initialize sess
	sess = tf.InteractiveSession(config=config)
	sess.run(tf.global_variables_initializer())

	first_performance = []
	last_performance = []
	tasks_acc = []

	for task in range(num_tasks_to_run):
		# print "Training task: ",task+1,"/",num_tasks_to_run
		print("\t task ", task+1)

		if task == 0:
			num_epochs_per_task = 50
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
			T_target_old = []
			clone_net(model_old,model)
			for pre_t in range(task):
				T_target_old.append(tf.matmul(model_old.fc1_drop, W_output[pre_t]) + b_output[pre_t])

			cost,cost_old,cost_new = lwf_criterion(t=task,targets_old=T_target_old,outputs=task_output,targets=y_,lamb=Lamb,T=2)

		train_op,learning_rate = optimizer(lr_start=learning_r, cost=cost, varlist=[model.params, W_output[task], b_output[task]],steps=lr_steps,epoches=epoches)
		sess.run(tf.variables_initializer([lr_steps]))
		for epoch in range(num_epochs_per_task):
			if epoch%5==0:
				print("\t Epoch ",epoch)
			for images, labels in yield_mb(tasks_train[task], tasks_train_label[task], batchsize=256, shuffle=True, one_hot=True):
				sess.run(train_op, feed_dict={x: images, y_: labels, keep_prob:0.5})
			
			# if task > 0:
			# 	print(sess.run(cost_new,feed_dict={x:batch[0], y_:batch[1]}))
			#print(sess.run(learning_rate))

		# Print test set accuracy to each task encountered so far
		avg_accuracy = 0.0
		for test_task in range(task+1):
			correct_prediction = tf.equal(tf.argmax(task_output[test_task], 1), tf.argmax(y_, 1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			test_data = tasks_test[test_task]

			acc = sess.run(accuracy, feed_dict={x: tasks_test[task], y_: label_to_one_hot(tasks_test_label[task], 10), keep_prob:1}) * 100.0
			avg_accuracy += acc

			if test_task == 0:
				first_performance.append(acc)
			if test_task == task:
				last_performance.append(acc)

			print("Task: ",test_task," \tAccuracy: ",acc)

		avg_accuracy = avg_accuracy/(task+1)
		print("Avg Perf: ",avg_accuracy)

	if avg_accuracy > best_accuracy:
		best_accuracy = avg_accuracy
		best_acc = tasks_acc
	print('best-aver-acc ',best_accuracy)
	per_avg_performance.append(avg_accuracy)

best_lamb = Lamb_set[per_avg_performance.index(max(per_avg_performance))]
print('best lamb is: ',best_lamb)

# save best model-test log
file= open('log.txt', 'w')
for fp in best_acc:
	file.write(str(fp))
	file.write('\n')
file.close()
ACC = best_accuracy

sess.close()

