# coding: utf-8

# automatically reload edited modules
import tensorflow as tf
import numpy as np
from model import *
import pickle


# train/compare vanilla sgd and ewc
def train_task(model, num_iter, disp_freq, trainset, testsets, x, y_,keep_prob,lams):    
    for l in range(len(lams)):
        # lams[l] sets weight on old task(s)
        model.restore(sess) # reassign optimal weights from previous training session
        if(lams[l] == 0):
            model.set_vanilla_loss()       
        # initialize test accuracy array for each task   
        test_accs = []
        for task in range(len(testsets)):
            test_accs.append(np.zeros(num_iter//disp_freq))
        # train on current task
        for iter in range(num_iter):
            for images,labels in yield_mb(trainset["data"],trainset["label"],batchsize=256,shuffle=True,one_hot=True):
                model.train_step.run(feed_dict={x: images, y_: labels,keep_prob:0.5})           
            if iter % disp_freq == 0:
               
                for task in range(len(testsets)):
                    feed_dict={x: testsets[task]["data"],y_:label_to_one_hot(testsets[task]["label"],2),
                               keep_prob:1.0}
                    test_accs[task][iter//disp_freq] = model.accuracy.eval(feed_dict=feed_dict)
                    
          
    return test_accs

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
y_ = tf.placeholder(tf.float32, shape=[None, 2])
keep_prob=tf.placeholder(tf.float32)
# instantiate new model
model = Model(x, y_,keep_prob) # simple 2-layer network
# initialize variables
sess.run(tf.global_variables_initializer())

# training 1st task
acc1=train_task(model,50,5, data01, [test01], x, y_,keep_prob,lams=[0])
print("1th task acc:",acc1[0][-1])
# save current optimal weights
model.star()

acc2=train_task(model,50,5,data23, [test23], x, y_,keep_prob, lams=[0])
print("2th task acc:",acc2[0][-1])
model.star()

acc3=train_task(model,50,5,data45, [test45], x, y_,keep_prob,lams=[0])
print("3th task acc:",acc3[0][-1])
model.star()

acc4=train_task(model,50,5,data67, [test67], x, y_,keep_prob,lams=[0])
print("4th task acc:",acc4[0][-1])
model.star()

acc5=train_task(model,50, 5, data89, [test01,test23,test45,test67,test89], x, y_,keep_prob,lams=[0])
print("1th task acc:",acc5[0][-1],"2th task acc:",acc5[1][-1],
      "3th task acc:",acc5[2][-1],"4th task acc:",acc5[3][-1],"5th task acc:",acc5[4][-1])

# evaluate
FD=(acc1[0][-1]-acc5[0][-1]+acc2[0][-1]-acc5[1][-1]+acc3[0][-1]-acc5[2][-1]+acc4[0][-1]-acc5[3][-1])/4 #计算ABCD四个任务的平均遗忘程度
print("FWT：",FD)
ACC5=(acc5[0][-1]+acc5[1][-1]+acc5[2][-1]+acc5[3][-1]+acc5[4][-1])/5
print("ACC：",ACC5)
model.star()
