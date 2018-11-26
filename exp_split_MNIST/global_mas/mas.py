# coding: utf-8

# automatically reload edited modules
import tensorflow as tf
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from IPython import display
from model import Model
import pickle

# mnist imshow convenience function
# input is a 1D array of length 784
def mnist_imshow(img):
    plt.imshow(img.reshape([28,28]), cmap="gray")
    plt.axis('off')

# return a new mnist dataset w/ pixels randomly permuted
def permute(mnist):
    perm_inds = list(range(mnist.train.images.shape[1]))
    np.random.shuffle(perm_inds)
    mnist2 = deepcopy(mnist)
    sets = ["train", "validation", "test"]
    for set_name in sets:
        this_set = getattr(mnist2, set_name) # shallow copy
        this_set._images = np.transpose(np.array([this_set.images[:,c] for c in perm_inds]))
    return mnist2

# classification accuracy plotting
def plot_test_acc(plot_handles, iter):
    plt.legend(handles=plot_handles, loc="center right")
    plt.xlabel("Iterations")
    plt.ylabel("Test Accuracy")
    plt.ylim(0,1)
    plt.savefig('iter.png')
    display.display(plt.gcf())
    display.clear_output(wait=True)

# train/compare vanilla sgd and ewc
def train_task1(model, num_iter, disp_freq, trainset, testsets, x,y1,keep_prob,lams):    
    for l in range(len(lams)):
        # lams[l] sets weight on old task(s)
        model.restore(sess) # reassign optimal weights from previous training session
        if(lams[l] == 0):
            model.set_vanilla_loss1()
        else:
            model.update_MAS_loss1(lams[l])
        # initialize test accuracy array for each task   
        test_accs = []
        for task in range(len(testsets)):
            test_accs.append(np.zeros(num_iter//disp_freq))
        # train on current task
        for iter in range(num_iter):
            for images,labels in yield_mb(trainset["data"],trainset["label"],batchsize=256,shuffle=True,one_hot=True):
                model.train_step1.run(feed_dict={x: images, y1: labels,keep_prob:0.5})            
            if iter % disp_freq == 0:                
                for task in range(len(testsets)):
                    feed_dict={x: testsets[task]["data"],y1:label_to_one_hot(testsets[task]["label"],2),
                               keep_prob:1.0}
                    test_accs[task][iter//disp_freq] = model.accuracy1.eval(feed_dict=feed_dict)                  
            
    return test_accs


def train_task2(model, num_iter, disp_freq, trainset, testsets, x,y2,keep_prob,lams):    
    for l in range(len(lams)):
        # lams[l] sets weight on old task(s)
        model.restore(sess) # reassign optimal weights from previous training session
        if(lams[l] == 0):
            model.set_vanilla_loss2()
        else:
            model.update_MAS_loss2(lams[l])
        # initialize test accuracy array for each task   
        test_accs = []
        for task in range(len(testsets)):
            test_accs.append(np.zeros(num_iter//disp_freq))
        # train on current task
        for iter in range(num_iter):
            for images,labels in yield_mb(trainset["data"],trainset["label"],batchsize=256,shuffle=True,one_hot=True):
                model.train_step2.run(feed_dict={x: images, y2: labels,keep_prob:0.5})            
            if iter % disp_freq == 0:                
                for task in range(len(testsets)):
                    feed_dict={x: testsets[task]["data"], y2:label_to_one_hot(testsets[task]["label"],2),
                               keep_prob:1.0}
                    test_accs[task][iter//disp_freq] = model.accuracy2.eval(feed_dict=feed_dict)                    
            
    return test_accs

def train_task3(model, num_iter, disp_freq, trainset, testsets, x,y3,keep_prob,lams):    
    for l in range(len(lams)):
        # lams[l] sets weight on old task(s)
        model.restore(sess) # reassign optimal weights from previous training session
        if(lams[l] == 0):
            model.set_vanilla_loss3()
        else:
            model.update_MAS_loss3(lams[l])
        # initialize test accuracy array for each task   
        test_accs = []
        for task in range(len(testsets)):
            test_accs.append(np.zeros(num_iter//disp_freq))
        # train on current task
        for iter in range(num_iter):
            for images,labels in yield_mb(trainset["data"],trainset["label"],batchsize=256,shuffle=True,one_hot=True):
                model.train_step3.run(feed_dict={x:images, y3: labels,keep_prob:0.5})            
            if iter % disp_freq == 0:                
                for task in range(len(testsets)):                    
                    feed_dict={x: testsets[task]["data"], y3:label_to_one_hot(testsets[task]["label"],2),
                               keep_prob:1.0}
                    test_accs[task][iter//disp_freq] = model.accuracy3.eval(feed_dict=feed_dict)                   
           
    return test_accs

def train_task4(model, num_iter, disp_freq, trainset, testsets, x,y4,keep_prob,lams):    
    for l in range(len(lams)):
        # lams[l] sets weight on old task(s)
        model.restore(sess) # reassign optimal weights from previous training session
        if(lams[l] == 0):
            model.set_vanilla_loss4()
        else:
            model.update_MAS_loss4(lams[l])
        # initialize test accuracy array for each task   
        test_accs = []
        for task in range(len(testsets)):
            test_accs.append(np.zeros(num_iter//disp_freq))
        # train on current task
        for iter in range(num_iter):
            for images,labels in yield_mb(trainset["data"],trainset["label"],batchsize=256,shuffle=True,one_hot=True):
                model.train_step4.run(feed_dict={x: images, y4: labels,keep_prob:0.5})         
            if iter % disp_freq == 0:              
                for task in range(len(testsets)):
                    feed_dict={x: testsets[task]["data"], y4:label_to_one_hot(testsets[task]["label"],2),
                               keep_prob:1.0}
                    test_accs[task][iter//disp_freq] = model.accuracy4.eval(feed_dict=feed_dict)                   
            
    return test_accs

def train_task5(model, num_iter, disp_freq, trainset, testsets, x,y1,y2,y3,y4,y5,keep_prob,lams):    
    for l in range(len(lams)):
        # lams[l] sets weight on old task(s)
        model.restore(sess) # reassign optimal weights from previous training session
        if(lams[l] == 0):
            model.set_vanilla_loss5()
        else:
            model.update_MAS_loss5(lams[l])
        # initialize test accuracy array for each task   
        test_accs = []
        for task in range(len(testsets)):
            test_accs.append(np.zeros(num_iter//disp_freq))
        # train on current task
        for iter in range(1,num_iter):
            for images,labels in yield_mb(trainset["data"],trainset["label"],batchsize=256,shuffle=True,one_hot=True):
                model.train_step5.run(feed_dict={x: images, y5: labels,keep_prob:0.5})
            if iter % disp_freq == 0:
                plt.subplot(1, len(lams), l+1)
                plots = []
                colors = ['r', 'b', 'g','y','k']
                for task in range(len(testsets)):                    
                    if task==0:   
                        feed_dict={x: testsets[task]["data"], y1:label_to_one_hot(testsets[task]["label"],2),keep_prob:1.0}
                        test_accs[task][iter//disp_freq] = model.accuracy1.eval(feed_dict=feed_dict)
                    elif task==1:
                        feed_dict={x: testsets[task]["data"], y2:label_to_one_hot(testsets[task]["label"],2),keep_prob:1.0}
                        test_accs[task][iter//disp_freq] = model.accuracy2.eval(feed_dict=feed_dict)
                    elif task==2:
                        feed_dict={x: testsets[task]["data"], y3:label_to_one_hot(testsets[task]["label"],2),keep_prob:1.0}
                        test_accs[task][iter//disp_freq] = model.accuracy3.eval(feed_dict=feed_dict)
                    elif task==3:
                        feed_dict={x: testsets[task]["data"], y4:label_to_one_hot(testsets[task]["label"],2),keep_prob:1.0}
                        test_accs[task][iter//disp_freq] = model.accuracy4.eval(feed_dict=feed_dict)
                    else:
                        feed_dict={x: testsets[task]["data"], y5:label_to_one_hot(testsets[task]["label"],2),keep_prob:1.0}
                        test_accs[task][iter//disp_freq] = model.accuracy5.eval(feed_dict=feed_dict)
                        
                    c = chr(ord('A') + task)
                    plot_h, = plt.plot(range(1,iter+2,disp_freq), test_accs[task][:iter//disp_freq+1], colors[task], label="task " + c)
                    plots.append(plot_h)                            
                plot_test_acc(plots, iter)
                if l == 0: 
                    plt.title("vanilla sgd")
                else:
                    plt.title("MAS")
                plt.gcf().set_size_inches(len(lams)*5, 3.5)
    return test_accs

# read split-MNIST
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
#data_task1
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
sess = tf.InteractiveSession()
# define input and target placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y1 = tf.placeholder(tf.float32, shape=[None,2])
y2 = tf.placeholder(tf.float32, shape=[None,2])
y3 = tf.placeholder(tf.float32, shape=[None,2])
y4 = tf.placeholder(tf.float32, shape=[None,2])
y5 = tf.placeholder(tf.float32, shape=[None,2])

keep_prob=tf.placeholder(tf.float32)
# instantiate new model
model = Model(x,y1,y2,y3,y4,y5,keep_prob) # simple 2-layer network
# initialize variables
sess.run(tf.global_variables_initializer())


# training 1st task
acc1=train_task1(model,50,5,data01,[test01], x,y1,keep_prob,lams=[0])
print("1th task acc:",acc1[0][-1])
# Fisher information
model.compute_omega1(sess,data01["data"], batch_size=1000,keep_prob=1.0)
model.save_omega()  # save recent task Fisher matrix
# save current optimal weights
model.star()

acc2=train_task2(model,50,5, data23, [test23], x,y2,keep_prob,lams=[0,0.1])
print("2th task acc:",acc2[0][-1])
model.compute_omega2(sess,data23["data"], batch_size=1000,keep_prob=1.0)
model.omega_accumulate()
model.save_omega()  # save recent task Fisher matrix
model.star()

acc3=train_task3(model,50,5,data45, [test45], x,y3,keep_prob,lams=[0,0.1])
print("3th task acc:",acc3[0][-1])
model.compute_omega3(sess,data45["data"], batch_size=1000,keep_prob=1.0)
model.omega_accumulate()
model.save_omega()
model.star()

acc4=train_task4(model,50,5, data67, [test67], x,y4,keep_prob,lams=[0,0.1])
print("4th task acc:",acc4[0][-1])
model.compute_omega4(sess,data67["data"],batch_size=1000,keep_prob=1.0)
model.omega_accumulate()
model.save_omega()
model.star()

acc5=train_task5(model,50,5, data89, [test01,test23,test45,test67,test89], x,y1,y2,y3,y4,y5,
                 keep_prob,lams=[0,0.1])
print("1th task acc:",acc5[0][-1],"2th task acc:",acc5[1][-1],
      "3th task acc:",acc5[2][-1],"4th task acc:",acc5[3][-1],"5th task acc:",acc5[4][-1])

###evaluation
FD=(acc1[0][-1]-acc5[0][-1]+acc2[0][-1]-acc5[1][-1]+acc3[0][-1]-acc5[2][-1]+acc4[0][-1]-acc5[3][-1])/4 #计算ABCD四个任务的平均遗忘程度
print("FWT：",FD)
ACC5=(acc5[0][-1]+acc5[1][-1]+acc5[2][-1]+acc5[3][-1]+acc5[4][-1])/5
print("ACC：",ACC5)
model.compute_omega5(sess,data89["data"],batch_size=1000,keep_prob=1.0)
model.omega_accumulate()
model.save_omega()
model.star()
