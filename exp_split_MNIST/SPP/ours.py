import numpy as np
from math import ceil
from model import *
import pickle





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

def compute_omega(sess, imgset, batch_size, param_im):
	Omega_M = []
	batch_num = ceil(len(imgset) / batch_size)
	for iter in range(batch_num):
		index = iter * batch_size
		Omega_M.append(sess.run(param_im, feed_dict={x: imgset[index:(index + batch_size - 1)]}))
	Omega_M = np.sum(Omega_M, 0) / batch_num

	return Omega_M


def optimizer(lr_start,cost,varlist,steps,epoches):
	learning_rate = tf.train.exponential_decay(lr_start, global_step=steps, decay_steps=epoches, decay_rate=0.9,staircase=True) # lr decay=1
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,global_step=steps, var_list=varlist)
	return train_step,learning_rate

# calculate params importance
def Cal_importance(varlist,output):
	importance = []
	probs = tf.nn.softmax(output)
	for v in range(len(varlist)):
		# # information entropy
		I = -tf.reduce_sum(probs * tf.log(probs))
		gradients = tf.gradients(I, varlist[v])# Gradient of the loss function for the current task
	# # cal importance --max（0,a）
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



## load data
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

tasks_train = [data01,data23,data45,data67,data89]
tasks_test = [test01,test23,test45,test67,test89]


# train and test setting
num_tasks_to_run = 5
num_epochs_per_task = 50
minibatch_size = 256
Lamb_set = [0.01,0.1,0.5,1,2,4,6,8,12] # hyper-parameter1 loss
epoches = 10*50000/minibatch_size #decay per 10 epoch
Beta = 1 # hyper-parameter2 sharpen
learning_r = 0.1

x = tf.placeholder(tf.float32, shape=[None, 784])
model = Model(x)
model_old = Model(x)

y_ = tf.placeholder(tf.float32, shape=[None, 2])
out_dim = int(y_.get_shape()[1])  # 10 for MNIST
lr_steps = tf.Variable(0)

# expand output layer
W_output = []
b_output = []
task_output = []
for task in range(num_tasks_to_run):
	W = weight_variable([32, out_dim])
	b = bias_variable([out_dim])
	W_output.append(W)
	b_output.append(b)
for task in range(num_tasks_to_run):
	output = tf.matmul(model.h2_drop, W_output[task]) + b_output[task]
	task_output.append(output)


config = tf.ConfigProto() # config
config.gpu_options.allow_growth = True


per_avg_performance = []
best_accuracy = 0.0
list_acc = []
for Lamb in Lamb_set:
	sess = tf.InteractiveSession(config=config)
	sess.run(tf.global_variables_initializer())

	Omega_v = []
	first_performance = []
	last_performance = []
	tasks_acc = []

	for task in range(num_tasks_to_run):
		# print "Training task: ",task+1,"/",num_tasks_to_run
		print("\t task ", task + 1)

		if task == 0:
			cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=task_output[task]))
		else:
			clone_net(model_old, model)
			cost_new = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=task_output[task]))
			cost = update_loss(pre_var=model_old.params, Omega=Omega_v, var=model.params, new_loss=cost_new, lamb=Lamb)
		train_op,learning_rate = optimizer(lr_start=learning_r, cost=cost, varlist=[model.params, W_output[task], b_output[task]],steps=lr_steps,epoches=epoches)
		sess.run(tf.variables_initializer([lr_steps]))

		for epoch in range(num_epochs_per_task):
			if epoch % 5 == 0:
				print("\t Epoch ", epoch)

			for images, labels in yield_mb(tasks_train[task]["data"], tasks_train[task]["label"], batchsize=256, shuffle=True, one_hot=True):
				sess.run(train_op, feed_dict={x: images, y_: labels})
		# calculate params importance
		param_importance = Cal_importance(varlist=model.params, output=task_output[task])

		# param_importance = Cal_importance(varlist=model.params, output=task_output[task],y_tgt=y_) # loss
		if task == 0:
			Omega_v = compute_omega(sess, tasks_train[task]["data"], batch_size=100,param_im=param_importance)

		else:
			for l in range(len(Omega_v)):
				Omega_v[l] = Omega_v[l] + compute_omega(sess, tasks_train[task]["data"], batch_size=100, param_im=param_importance)[l]

			# sharpen
			Omega_v = im_sharpen(Omega_v, sharpen=False, beta=Beta)
		# Print test set accuracy to each task encountered so far
		avg_accuracy = 0.0
		for test_task in range(task + 1):
			correct_prediction = tf.equal(tf.argmax(task_output[test_task], 1), tf.argmax(y_, 1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			test_data = tasks_test[test_task]

			acc = sess.run(accuracy, feed_dict={x: test_data["data"], y_: label_to_one_hot(test_data["label"],2)}) * 100.0
			avg_accuracy += acc

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
print(list_acc)
# save best model-test log
file= open('log-ours.txt', 'w')
for fp in best_acc:
	file.write(str(fp))
	file.write('\n')
file.write(str(per_avg_performance))
file.close()


sess.close()
