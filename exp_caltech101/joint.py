import tensorflow as tf
import numpy as np
from math import ceil
import tensorflow.contrib as tf_contrib
import random
import os
import scipy.misc as misc
from skimage import color
import math
def load_images_path(dataset_path):
    classes = sorted(os.listdir(dataset_path))
    image=[]
    label=[]
    class_num=len(classes)
    for i in range(class_num):
        class_path = os.path.join(dataset_path, classes[i])
        imgs = sorted(os.listdir(class_path))
        img_num = len(imgs)
        for j in range(img_num):
            img = os.path.join(class_path, imgs[j])
            image.append(img)
            label.append(i)
    return image,label
def sub_pix_mean(imgs_arr,train_mean_vector):
    imgs_arr=np.array(imgs_arr)
    images=np.reshape(imgs_arr,[imgs_arr.shape[0], -1])
    images=(images-train_mean_vector)/255.0
    images=np.reshape(images,[images.shape[0],224,224,3])
    return images
def get_train_pix_mean(image):
    imgs_arr = np.array(image)
    images = np.reshape(imgs_arr, [imgs_arr.shape[0], -1])
    mean_image = np.mean(images)
    #std_image = np.std(images)
    return mean_image
def mean_vector(images):
    meanvector = np.zeros((1, 3))
    meanvector[:, 0] = np.mean(images[:, :, :, 0])
    meanvector[:, 1] = np.mean(images[:, :, :, 1])
    meanvector[:, 2] = np.mean(images[:, :, :, 2])
    return meanvector
def onehot(label,class_num):
    labels = np.zeros([len(label), class_num])
    labels[np.arange(len(label)), label] = 1
    labels = np.reshape(labels, [-1, class_num])
    return labels
def shuffle(train_samples,train_x,train_y):
    # shuffle the dataset
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    # permutation = np.random.permutation(train_samples)
    # x_train =train_x[permutation,:,:,:]
    # y_train =train_y[permutation]

    perm = np.arange(train_samples)
    random.shuffle(perm)
    x_train= train_x[np.array(perm)]
    y_train = train_y[np.array(perm)]
    return x_train,y_train
def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch
def read_images(batch_image):
    batch_x=[]
    for img in batch_image:
        img_arr=misc.imread(img)
        resize_image = misc.imresize(img_arr, [224, 224, 3])
        if resize_image.shape != (224, 224, 3):
            resize_image = color.grey2rgb(resize_image)/255.0
        else:
            resize_image = resize_image/255.0
        batch_x.append(resize_image)
    return np.array(batch_x)
def load_task(train_path,test_path):
    task=[]
    train_x, train_y = load_images_path(train_path)
    test_x, test_y = load_images_path(test_path)
    task.append(train_x)
    task.append(train_y)
    task.append(test_x)
    task.append(test_y)
    return task
def residual_block_first(x, w1, w2,w3, b1, b2, strides1,strides2, pad="SAME"):
    in_channel = x.get_shape().as_list()[-1]
    # Shortcut connection
    if in_channel == w2.get_shape().as_list()[-1]:
        if strides1[1] == 1:
            shortcut = tf.identity(x)
        else:
            shortcut = tf.nn.max_pool(x, [1,2,2,1], [1,2,2,1], 'VALID')
    else:  # 输入和输出通道数不相同的情况
        shortcut = tf.nn.conv2d(x, w3,strides=[1,2,2,1],padding="SAME",name='shortcut')
        shortcut=batch_normlize(shortcut)

    conv_1 = tf.nn.conv2d(x, w1, strides=strides1, padding=pad) + b1
    bn_1 = tf.nn.relu(batch_normlize(conv_1))

    conv_2 = tf.nn.conv2d(bn_1, w2, strides=strides2, padding=pad) + b2
    bn_2 = batch_normlize(conv_2)
    out = tf.nn.relu(shortcut + bn_2)
    return out
def residual_block(x, w1, w2, b1, b2, strides1,strides2, pad="SAME"):
    shorcut = x
    conv_1 = tf.nn.conv2d(x, w1, strides=strides1, padding=pad) + b1
    bn_1 = tf.nn.relu(batch_normlize(conv_1))
    conv_2 = tf.nn.conv2d(bn_1, w2, strides=strides2, padding=pad) + b2
    bn_2 = batch_normlize(conv_2)
    out = tf.nn.relu(shorcut + bn_2)
    return out
def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    return gap
def weight_init(shape,name):
    with tf.name_scope('weights'):
        return tf.get_variable(name=name,shape=shape,initializer=tf.contrib.layers.xavier_initializer())
def bias_variable(shape):
    with tf.name_scope('bias'):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

def batch_normlize(x):
    return tf_contrib.layers.batch_norm(x, decay=0.9, epsilon=1e-05, center=True, scale=True, updates_collections=None)
class Model():
    def __init__(self,x):
        self.x = x  # input placeholder

        ##conv1-bn-pool
        self.w1 = weight_init(shape=[7, 7, 3, 64],name='w1')
        self.b1 = bias_variable(shape=[64])
        conv1 = tf.nn.conv2d(x, self.w1, strides=[1, 2, 2, 1], padding="SAME") + self.b1
        conv1_bn = tf.nn.relu(batch_normlize(conv1))
        pool1 = tf.nn.max_pool(conv1_bn, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        # conv2_x
        # conv2_1
        self.w2 = weight_init(shape=[3, 3, 64, 64],name='w2')
        self.b2 = bias_variable(shape=[64])
        self.w3 = weight_init(shape=[3, 3, 64, 64],name='w3')
        self.b3 = bias_variable(shape=[64])
        conv2_1 = residual_block(pool1, self.w2, self.w3, self.b2,self.b3, strides1=[1, 1, 1, 1], strides2=[1, 1, 1, 1])

        ##conv2_2
        self.w4 = weight_init(shape=[3, 3, 64, 64],name='w4')
        self.b4 = bias_variable(shape=[64])
        self.w5 = weight_init(shape=[3, 3, 64, 64],name='w5')
        self.b5 = bias_variable(shape=[64])
        conv2_2 = residual_block(conv2_1, self.w4, self.w5, self.b4, self.b5, strides1=[1, 1, 1, 1], strides2=[1, 1, 1, 1])

        ##conv3_1
        self.w6 = weight_init(shape=[1, 1, 64, 128],name='w6')
        self.w7 = weight_init(shape=[3, 3, 64, 128],name='w7')
        self.b7 = bias_variable(shape=[128])
        self.w8 = weight_init(shape=[3, 3, 128, 128],name='w8')
        self.b8 = bias_variable(shape=[128])
        conv3_1 = residual_block_first(conv2_2, self.w7, self.w8, self.w6, self.b7, self.b8, strides1=[1, 2, 2, 1], strides2=[1, 1, 1, 1])

        ##conv3_2
        self.w9 = weight_init(shape=[3, 3, 128, 128],name='w9')
        self.b9 = bias_variable(shape=[128])
        self.w10 = weight_init(shape=[3, 3, 128, 128],name='w10')
        self.b10 = bias_variable(shape=[128])
        conv3_2 = residual_block(conv3_1, self.w9, self.w10, self.b9,self.b10, strides1=[1, 1, 1, 1], strides2=[1, 1, 1, 1])

        # conv4_1
        self.w11 = weight_init(shape=[1, 1, 128, 256],name='w11')
        self.w12 = weight_init(shape=[3, 3, 128, 256],name='w12')
        self.b12 = bias_variable(shape=[256])
        self.w13 = weight_init(shape=[3, 3, 256, 256],name='w13')
        self.b13 = bias_variable(shape=[256])
        conv4_1 = residual_block_first(conv3_2, self.w12, self.w13, self.w11,self.b12, self.b13, strides1=[1, 2, 2, 1], strides2=[1, 1, 1, 1])

        ##conv4_2
        self.w14 = weight_init(shape=[3, 3, 256, 256],name='w14')
        self.b14 = bias_variable(shape=[256])
        self.w15 = weight_init(shape=[3, 3, 256, 256],name='w15')
        self.b15 = bias_variable(shape=[256])
        conv4_2 = residual_block(conv4_1,self.w14, self.w15, self.b14,self.b15, strides1=[1, 1, 1, 1], strides2=[1, 1, 1, 1])

        ##conv5_1
        self.w16 = weight_init(shape=[1, 1, 256, 512],name='w16')
        self.w17 = weight_init(shape=[3, 3, 256, 512],name='w17')
        self.b17 = bias_variable(shape=[512])
        self.w18 = weight_init(shape=[3, 3, 512, 512],name='w18')
        self.b18 = bias_variable(shape=[512])
        conv5_1 = residual_block_first(conv4_2, self.w17, self.w18, self.w16, self.b17, self.b18, strides1=[1, 2, 2, 1], strides2=[1, 1, 1, 1])

        ##conv5_2
        self.w19 = weight_init(shape=[3, 3, 512, 512],name='w19')
        self.b19 = bias_variable(shape=[512])
        self.w20 = weight_init(shape=[3, 3, 512, 512],name='w20')
        self.b20 = bias_variable(shape=[512])
        conv5_2 = residual_block(conv5_1, self.w19, self.w20, self.b19, self.b20, strides1=[1, 1, 1, 1], strides2=[1, 1, 1, 1])

        ##avg_pool-flatten-fc
        # pool_6 = tf.nn.avg_pool(conv5_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        pool_6 = global_avg_pooling(conv5_2)
        dim = pool_6.get_shape().as_list()
        flat_dim = dim[1] * dim[2] * dim[3]
        flat_6 = tf.reshape(pool_6, [-1, flat_dim])

        self.w21 = weight_init(shape=[flat_dim,2048],name='w21')
        self.b21 = bias_variable(shape=[2048])
        fc1 = tf.nn.relu(tf.matmul(flat_6, self.w21) + self.b21)
        self.drop1=tf.nn.dropout(fc1,keep_prob=0.5)
        self.params=[self.w1,self.b1,self.w2,self.b2,self.w3,self.b3,self.w4,self.b4,self.w5,self.b5,self.w6,self.w7,
                     self.b7,self.w8,self.b8,self.w9,self.b9,self.w10,self.b10,self.w11,self.w12,self.b12,self.w13,
                     self.b13,self.w14,self.b14,self.w15,self.b15,self.w16,self.w17,self.b17,self.w18,self.b18,self.w19,
                     self.b19,self.w20,self.b20,self.w21,self.b21]
        return

def optimizer(lr_start, cost, varlist, steps, epoches):
	learning_rate = tf.train.exponential_decay(lr_start, global_step=steps, decay_steps=epoches, decay_rate=0.9,staircase=True)  # lr decay=1
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=steps, var_list=varlist)
	return train_step, learning_rate

def list_array(list):
	a = list[0]
	for i in range(1, len(list)):
		a = np.vstack((a,list[i]))
	return a

x = tf.placeholder(tf.float32, shape=[None, 224,224,3])
model = Model(x)

# train and test
num_tasks_to_run = 4
num_epochs_per_task = 500
minibatch_size = 128
epoches = 6000
learning_lr = 0.001


train_1_path=r"./Caltech101_split_200/task_1/train"
test_1_path=r"./Caltech101_split_200/task_1/test"
train_2_path=r"./Caltech101_split_200/task_2/train"
test_2_path=r"./Caltech101_split_200/task_2/test"
train_3_path=r"./Caltech101_split_200/task_3/train"
test_3_path=r"./Caltech101_split_200/task_3/test"
train_4_path=r"./Caltech101_split_200/task_4/train"
test_4_path=r"./Caltech101_split_200/task_4/test"
task1=load_task(train_1_path,test_1_path)
task2=load_task(train_2_path,test_2_path)
task3=load_task(train_3_path,test_3_path)
task4=load_task(train_4_path,test_4_path)
# Generate the tasks specifications as a list of random permutations of the input pixels.
task_permutation = [task1,task2,task3,task4]

label_list_train = []
img_list_train = []
label_list_test = []
img_list_test = []
for j in range(len(task_permutation)):
	label_list_train.append(task_permutation[j][1])
	img_list_train.append(task_permutation[j][0])
	label_list_test.append(task_permutation[j][3])
	img_list_test.append(task_permutation[j][2])

def one_hot_expand(label_list_train):
    shape = len(label_list_train[0])
    label_list_train[0] = np.hstack((onehot(label_list_train[0],30),np.zeros([shape,72])))
    shape = len(label_list_train[1])
    label_list_train[1] = np.hstack((onehot(label_list_train[1],25),np.zeros([shape,47])))
    label_list_train[1] = np.hstack((np.zeros([shape,30]),label_list_train[1]))
    shape = len(label_list_train[2])
    label_list_train[2] = np.hstack((onehot(label_list_train[2],25),np.zeros([shape,22])))
    label_list_train[2] = np.hstack((np.zeros([shape,55]),label_list_train[2]))
    shape = len(label_list_train[3])
    label_list_train[3] = np.hstack((np.zeros([shape,80]),onehot(label_list_train[3],22)))
    return label_list_train
def merge_train(img_list_train):
    img=[]
    for i in img_list_train:
        img.extend(i)
    return img
label_list_train = list_array(one_hot_expand(label_list_train))
label_list_test = one_hot_expand(label_list_test)
img_list_train = merge_train(img_list_train)
img_list_test = img_list_test

y_ = tf.placeholder(tf.float32, shape=[None, 102])
out_dim = int(y_.get_shape()[1])
lr_steps = tf.Variable(0)

# expand output layer
W_output = weight_init([2048, out_dim],name='out_joint')
b_output = bias_variable([out_dim])
task_output = tf.matmul(model.drop1, W_output) + b_output

# loss function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=task_output))
train_op, learning_rate = optimizer(lr_start=learning_lr, cost=cost, varlist=[model.params, W_output, b_output],steps=lr_steps, epoches=epoches)

correct_prediction = tf.equal(tf.argmax(task_output, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
## Initialize session
config = tf.ConfigProto()

config.gpu_options.allow_growth = True

sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())

stop_loss = 1000000
total_count = 0
for epoch in range(1,num_epochs_per_task+1):
    print("\t Epoch ", epoch)
    loss_all, acc_all, lr_all = [], [], []
    train_img, train_lab = shuffle(len(img_list_train), img_list_train,label_list_train)
    for i in range(int(len(img_list_train) / minibatch_size) + 1):
        batch_x = train_img[i * minibatch_size:(i + 1) * minibatch_size]
        batch_y = train_lab[i * minibatch_size:(i + 1) * minibatch_size]
        train_batch_x = read_images(batch_x)
        train_batch_img = _random_flip_leftright(train_batch_x)
        sess.run(train_op, feed_dict={x: train_batch_img, y_:batch_y})

        loss_1, acc_1, lr_1 = sess.run([cost, accuracy, learning_rate],
                                        feed_dict={x: train_batch_img, y_: batch_y})
        loss_all.append(loss_1)
        acc_all.append(acc_1)
        lr_all.append(lr_1)

    mean_loss = np.mean(loss_all)
    mean_acc = np.mean(acc_all)
    mean_lr = np.mean(lr_all)
    if epoch % 5 == 0:  # 5次打印下loss
        print("Epoch " + str(epoch) + ", Minibatch Loss=" + \
            str(mean_loss) + ", Training Accuracy= " + \
            str(mean_acc) + ", learning_rate:", str(mean_lr))
    if mean_loss < stop_loss:
        stop_loss = mean_loss
        total_count = 0
    else:
        total_count += 1
    if total_count >= 10:
        break

tasks_acc = []
avg_accuracy = 0.0
for test_task in range(num_tasks_to_run):
    #acc = sess.run(accuracy, feed_dict={x: img_list_test[test_task], y_: label_list_test[test_task]})
    test_x, test_y = img_list_test[test_task],label_list_test[test_task]
    test_img, test_lab = shuffle(len(test_x), test_x, test_y)
    iters = len(test_x) // minibatch_size
    te_acc = []
    for k in range(0, iters):
        test_batch_x = test_img[k * minibatch_size:(k + 1) * minibatch_size]
        test_batch_y = test_lab[k * minibatch_size:(k + 1) * minibatch_size]
        test_batch_images = read_images(test_batch_x)
        acc_ = sess.run(accuracy, feed_dict={x: test_batch_images, y_: test_batch_y})
        te_acc.append(acc_)
    test_avg = np.mean(te_acc)

    avg_accuracy += test_avg
    tasks_acc.append(test_avg)
    print("Task: ", test_task, " \tAccuracy: ", test_avg)

avg_accuracy = avg_accuracy / 4
print("Avg Perf: ", avg_accuracy)


# save best model-test log
best_acc = tasks_acc
file = open('log-join.txt', 'w')
for fp in best_acc:
	file.write(str(fp))
	file.write('\n')
file.write(str(avg_accuracy))
file.close()
sess.close()