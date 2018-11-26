# coding: utf-8

import numpy as np
from model import *
import pickle
import cv2
import struct
from copy import deepcopy
import gc
from scipy.io import loadmat as load
import os


# train/compare vanilla sgd and ewc
def train_task1(model, num_iter, disp_freq, trainset ,trainlabels, testsets, testlabels, x,y1,keep_prob,lams):    
    for l in range(len(lams)):
        # lams[l] sets weight on old task(s)
        if(lams[l] == 0):
            model.set_vanilla_loss1()       
        # initialize test accuracy array for each task   
        test_accs = []
        for task in range(len(testsets)):
            test_accs.append(np.zeros(num_iter//disp_freq))
        # train on current task
        for iter in range(num_iter):
            for images,labels in yield_mb(trainset[task], trainlabels[task], batchsize=256, shuffle=True, one_hot=True):
                model.train_step1.run(feed_dict={x: images, y1: labels, keep_prob:0.5})           
            if iter % disp_freq == 0:
                for task in range(len(testsets)):
                    feed_dict={x: testsets[task], y1: label_to_one_hot(testlabels[task], 10), keep_prob:1}
                    test_accs[task][iter//disp_freq] = model.accuracy1.eval(feed_dict=feed_dict)
    return test_accs


def train_task2(model, num_iter, disp_freq, trainset ,trainlabels,testsets, testlabels, x,y2,keep_prob,lams):    
    for l in range(len(lams)):
        # lams[l] sets weight on old task(s)
        if(lams[l] == 0):
            model.set_vanilla_loss2()       
        # initialize test accuracy array for each task   
        test_accs = []
        for task in range(len(testsets)):
            test_accs.append(np.zeros(num_iter//disp_freq))
        # train on current task
        for iter in range(num_iter):
            for images,labels in yield_mb(trainset[task], trainlabels[task], batchsize=256, shuffle=True, one_hot=True):
                model.train_step2.run(feed_dict={x: images, y2: labels, keep_prob:0.5})           
            if iter % disp_freq == 0:
                for task in range(len(testsets)):
                    feed_dict={x: testsets[task], y2: label_to_one_hot(testlabels[task], 10), keep_prob:1}
                    test_accs[task][iter//disp_freq] = model.accuracy2.eval(feed_dict=feed_dict)
    return test_accs

def train_task3(model, num_iter, disp_freq, trainset ,trainlabels, testsets, testlabels, x,y3,keep_prob,lams):    
    for l in range(len(lams)):
       # lams[l] sets weight on old task(s)
        if(lams[l] == 0):
            model.set_vanilla_loss3()       
        # initialize test accuracy array for each task   
        test_accs = []
        for task in range(len(testsets)):
            test_accs.append(np.zeros(num_iter//disp_freq))
        # train on current task
        for iter in range(num_iter):
            for images,labels in yield_mb(trainset[task], trainlabels[task], batchsize=256, shuffle=True, one_hot=True):
                model.train_step3.run(feed_dict={x: images, y3: labels, keep_prob:0.5})           
            if iter % disp_freq == 0:
                for task in range(len(testsets)):
                    feed_dict={x: testsets[task], y3: label_to_one_hot(testlabels[task], 10), keep_prob:1}
                    test_accs[task][iter//disp_freq] = model.accuracy3.eval(feed_dict=feed_dict)               
           
    return test_accs

def train_task4(model, num_iter, disp_freq, trainset ,trainlabels, testsets, testlabels, x,y4,keep_prob,lams):    
    for l in range(len(lams)):
        # lams[l] sets weight on old task(s)
        if(lams[l] == 0):
            model.set_vanilla_loss4()       
        # initialize test accuracy array for each task   
        test_accs = []
        for task in range(len(testsets)):
            test_accs.append(np.zeros(num_iter//disp_freq))
        # train on current task
        for iter in range(num_iter):
            for images,labels in yield_mb(trainset[task], trainlabels[task], batchsize=256, shuffle=True, one_hot=True):
                model.train_step4.run(feed_dict={x: images, y4: labels, keep_prob:0.5})           
            if iter % disp_freq == 0:
                for task in range(len(testsets)):
                    feed_dict={x: testsets[task], y4: label_to_one_hot(testlabels[task], 10), keep_prob:1}
                    test_accs[task][iter//disp_freq] = model.accuracy4.eval(feed_dict=feed_dict)                
            
    return test_accs

def train_task5(model, num_iter, disp_freq, trainset ,trainlabels, testsets, testlabels, x,y1,y2,y3,y4,y5,keep_prob,lams):    
    for l in range(len(lams)):
        # lams[l] sets weight on old task(s)
        model.restore(sess) # reassign optimal weights from previous training session
        if(lams[l] == 0):
            model.set_vanilla_loss5()
        else:
            model.update_ewc_loss5(lams[l])
        # initialize test accuracy array for each task   
        test_accs = []
        for task in range(len(testsets)):
            test_accs.append(np.zeros(num_iter//disp_freq))
        # train on current task
        for iter in range(1,num_iter):
            for images,labels in yield_mb(trainset[task], trainlabels[task], batchsize=256, shuffle=True, one_hot=True):
                model.train_step5.run(feed_dict={x: images, y5: labels,keep_prob:0.5})
            if iter % disp_freq == 0:
                for task in range(len(testsets)):                    
                    if task==0:   
                        feed_dict={x: testsets[task], y1:label_to_one_hot(testlabels[task],10),keep_prob:1.0}
                        test_accs[task][iter//disp_freq] = model.accuracy1.eval(feed_dict=feed_dict)
                    elif task==1:
                        feed_dict={x: testsets[task], y1:label_to_one_hot(testlabels[task],10),keep_prob:1.0}
                        test_accs[task][iter//disp_freq] = model.accuracy2.eval(feed_dict=feed_dict)
                    elif task==2:
                        feed_dict={x: testsets[task], y1:label_to_one_hot(testlabels[task],10),keep_prob:1.0}
                        test_accs[task][iter//disp_freq] = model.accuracy3.eval(feed_dict=feed_dict)
                    elif task==3:
                        feed_dict={x: testsets[task], y1:label_to_one_hot(testlabels[task],10),keep_prob:1.0}
                        test_accs[task][iter//disp_freq] = model.accuracy4.eval(feed_dict=feed_dict)
                    else:
                        feed_dict={x: testsets[task], y1:label_to_one_hot(testlabels[task],10),keep_prob:1.0}
                        test_accs[task][iter//disp_freq] = model.accuracy5.eval(feed_dict=feed_dict)
                        
    return test_accs


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
		magic, n = struct.unpack('>II', lbpath.read(8))# 'I'表示一个无符号整数，大小为四个字节 # '>II'表示读取两个无符号整数，即8个字节 
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
    x_train =  x_train/255.0
    x_test = x_test/255.0
    # Reshape
    x_train = x_train.reshape(-1, 32, 32, 3)
    x_test = x_test.reshape(-1, 32, 32, 3)
	
    #print(x_train.shape[0])
    #x_train = data_augmentation(x_train)
    #print(x_train.shape[0])

    #y_train = np.concatenate((y_train, y_train, y_train), axis=0)
    #print(y_train.shape[0])
	
    assert len(x_train) == len(y_train)
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
    x_train = image_crop(x_train, shape)

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
    x_train = image_crop(x_train, shape)
   # x_train = data_augmentation(x_train)
    #print(x_train.shape)
    #y_train = np.concatenate((y_train, y_train, y_train), axis=0)
   # print(y_train.shape)
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
    #assert len(X) == len(y)
    if shuffle:
        X, y = shuffle_data(X, y)
    if one_hot:
        y=label_to_one_hot(y, 10)
    # Only complete batches are submitted
    for i in range(len(X) // batchsize):
        yield X[i * batchsize:(i + 1) * batchsize], y[i * batchsize:(i + 1) * batchsize]


x0, tx0, y0, ty0 = MNIST_for_library()
x1, tx1, y1, ty1 = notMNIST_for_library()
x2, tx2, y2, ty2 = svhn_for_library()
x3, tx3, y3, ty3 = stl_for_library()
x4, tx4, y4, ty4 = cifar_for_library()


sess = tf.InteractiveSession()
# define input and target placeholders
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y1 = tf.placeholder(tf.float32, shape=[None,10])
y2 = tf.placeholder(tf.float32, shape=[None,10])
y3 = tf.placeholder(tf.float32, shape=[None,10])
y4 = tf.placeholder(tf.float32, shape=[None,10])
y5 = tf.placeholder(tf.float32, shape=[None,10])

keep_prob=tf.placeholder(tf.float32)
# instantiate new model
model = Model(x,y1,y2,y3,y4,y5,keep_prob) # simple 2-layer network
# initialize variables
sess.run(tf.global_variables_initializer())

# training 1st task
acc1=train_task1(model,100,5, [x0], [y0], [tx0], [ty0],x, y1,keep_prob,lams=[0])
print("1th task acc:",acc1[0][-1])
# save current optimal weights
model.star()

acc2=train_task2(model,100,5,[x1], [y1], [tx1], [ty1],x, y2,keep_prob,lams=[0])
print("2th task acc:",acc2[0][-1])
model.star()

acc3=train_task3(model,100,5,[x2], [y2], [tx2], [ty2],x, y3,keep_prob,lams=[0])
print("3th task acc:",acc3[0][-1])
model.star()

acc4=train_task4(model,100,5 ,[x3], [y3], [tx3], [ty3], x, y4,keep_prob,lams=[0])
print("4th task acc:",acc4[0][-1])
model.star()


acc5=train_task5(model,100, 5, [x4], [y4], [tx0, tx1, tx2, tx3, tx4], [ty0, ty1, ty2, ty3, ty4], x,y1,y2,y3,y4, y5, keep_prob,lams=[0])
print("1th task acc:",acc5[0][-1],"2th task acc:",acc5[1][-1],
      "3th task acc:",acc5[2][-1],"4th task acc:",acc5[3][-1],"5th task acc:",acc5[4][-1])

FD=(acc1[0][-1]-acc5[0][-1]+acc2[0][-1]-acc5[1][-1]+acc3[0][-1]-acc5[2][-1]+acc4[0][-1]-acc5[3][-1])/4 #计算ABCD四个任务的平均遗忘程度
print("FWT：",FD)
ACC5=(acc5[0][-1]+acc5[1][-1]+acc5[2][-1]+acc5[3][-1]+acc5[4][-1])/5
print("ACC：",ACC5)
model.star()





