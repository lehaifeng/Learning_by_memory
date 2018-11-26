import tensorflow as tf
import numpy as np
import pickle
from model import *
# load data

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



x = tf.placeholder(tf.float32,shape=[None,784])
model = Model(x,keep_prob=0.5)
# x_clone = tf.placeholder(tf.float32,shape=[None,784])
model_old = Model(x,keep_prob=0.5)
# train and test
num_tasks_to_run = 5
num_epochs_per_task = 50
minibatch_size = 256
learning_r = 0.01
epoches = 10*50000/minibatch_size #decay per 10 epoch
Beta = 1 # hyper-parameter2 sharpen
Lamb_set = [0.1,0.5,1,2,4,6,8,12]# lambda hyper-parameter search

# Generate the tasks specifications as a list of random permutations of the input pixels.

W_output = []
b_output = []
task_output = []

y_ = tf.placeholder(tf.float32, shape=[None, 2])
out_dim = int(y_.get_shape()[1]) # 10 for MNIST
lr_steps = tf.Variable(0)

for task in range(num_tasks_to_run):
	W = weight_variable([32,out_dim])
	b = bias_variable([out_dim])
	W_output.append(W)
	b_output.append(b)
for task in range(num_tasks_to_run):
	output = tf.matmul(model.h2_drop,W_output[task]) + b_output[task]
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
			cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=task_output[task]))
		else:
			T_target_old = []
			clone_net(model_old,model)
			for pre_t in range(task):
				T_target_old.append(tf.matmul(model_old.h2_drop,W_output[pre_t]) + b_output[pre_t])

			cost,cost_old,cost_new = lwf_criterion(t=task,targets_old=T_target_old,outputs=task_output,targets=y_,lamb=Lamb,T=2)

		train_op,learning_rate = optimizer(lr_start=learning_r, cost=cost, varlist=[model.params, W_output[task], b_output[task]],steps=lr_steps,epoches=epoches)
		sess.run(tf.variables_initializer([lr_steps]))
		for epoch in range(num_epochs_per_task):
			if epoch%5==0:
				print("\t Epoch ",epoch)
			for images, labels in yield_mb(tasks_train[task]["data"], tasks_train[task]["label"], batchsize=256, shuffle=True, one_hot=True):
    				sess.run(train_op, feed_dict={x: images, y_: labels})
		
		# Print test set accuracy to each task encountered so far
		avg_accuracy = 0.0
		for test_task in range(task+1):
			correct_prediction = tf.equal(tf.argmax(task_output[test_task], 1), tf.argmax(y_, 1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
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

