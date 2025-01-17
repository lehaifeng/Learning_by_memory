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
    else:
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
    def star(self):
        # used for saving optimal weights after most recent task training
        self.star_vars = []
        for v in range(len(self.params)):
            self.star_vars.append(self.params[v].eval())

    def restore(self, sess):
        # reassign optimal weights for latest task
        for v in range(len(self.params)):
            sess.run(self.params[v].assign(self.star_vars[v]))
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
x = tf.placeholder(tf.float32, shape=[None, 224,224,3])
model = Model(x)
# train and test
num_tasks_to_run = 4
num_epochs_per_task = 300
minibatch_size = 128
Lamb_set = [6] # hyper-parameter1 loss
epoches = 3000
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
y1 = tf.placeholder(tf.float32, shape=[None,30])
y2 = tf.placeholder(tf.float32, shape=[None,25])
y3 = tf.placeholder(tf.float32, shape=[None,25])
y4 = tf.placeholder(tf.float32, shape=[None,22])
out_dim1 = int(y1.get_shape()[1])
out_dim2 = int(y2.get_shape()[1])
out_dim3 = int(y3.get_shape()[1])
out_dim4 = int(y4.get_shape()[1])
y_=[y1,y2,y3,y4]
out_dim=[out_dim1,out_dim2,out_dim3,out_dim4]
lr_steps = tf.Variable(0)

# Generate the tasks specifications as a list of random permutations of the input pixels.
W_output = []
b_output = []
task_output = []
name=['out1','out2','out3','out4']


for task in range(num_tasks_to_run):
    W = weight_init(shape=[2048, out_dim[task]],name=name[task])
    b = bias_variable([out_dim[task]])
    W_output.append(W)
    b_output.append(b)
for task in range(num_tasks_to_run):
    output = tf.matmul(model.drop1, W_output[task]) + b_output[task]
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
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_[task], logits=task_output[task]))
        else:
            T_target_old = []
            model.restore(sess)
            for pre_t in range(task):
                T_target_old.append(tf.matmul(model.drop1,W_output[pre_t]) + b_output[pre_t])
            cost,cost_old,cost_new = lwf_criterion(t=task,targets_old=T_target_old,outputs=task_output,targets=y_[task],lamb=Lamb,T=2)

        train_op,learning_rate = optimizer(lr_start=learning_lr, cost=cost, varlist=[model.params, W_output[task], b_output[task]],steps=lr_steps,epoches=epoches)
        correct_t = tf.equal(tf.argmax(task_output[task], 1), tf.argmax(y_[task], 1))
        acc_t = tf.reduce_mean(tf.cast(correct_t, tf.float32))
        sess.run(tf.variables_initializer([lr_steps]))
        stop_loss = 1000000
        total_count = 0
        for epoch in range(1, num_epochs_per_task + 1):
            print("\t Epoch ", epoch)
            loss_all, acc_all, lr_all = [], [], []
            train_img, train_lab = shuffle(len(task_permutation[task][0]), task_permutation[task][0], task_permutation[task][1])
            for i in range(int(len(task_permutation[task][0]) / minibatch_size) + 1):
                batch_x = train_img[i * minibatch_size:(i + 1) * minibatch_size]
                batch_y = train_lab[i * minibatch_size:(i + 1) * minibatch_size]
                train_batch_x = read_images(batch_x)
                train_batch_img = _random_flip_leftright(train_batch_x)
                train_batch_lab = onehot(batch_y, out_dim[task])
                sess.run(train_op, feed_dict={x: train_batch_img, y_[task]: train_batch_lab})

                loss_1, acc_1, lr_1 = sess.run([cost, acc_t, learning_rate],
                                                feed_dict={x: train_batch_img, y_[task]: train_batch_lab})
                loss_all.append(loss_1)
                acc_all.append(acc_1)
                lr_all.append(lr_1)

            mean_loss = np.mean(loss_all)
            mean_acc = np.mean(acc_all)
            mean_lr = np.mean(lr_all)
            if epoch % 5 == 0:
                print("Epoch " + str(epoch) + ", Minibatch Loss=" + \
                      str(mean_loss) + ", Training Accuracy= " + \
                      str(mean_acc) + ", learning_rate:", str(mean_lr))
            if mean_loss<stop_loss:
                stop_loss=mean_loss
                total_count=0
            else:
                total_count+=1
            if total_count>=10:
                break

        model.star()
        # Print test set accuracy to each task encountered so far
        avg_accuracy = 0.0
        for test_task in range(task + 1):
            correct_prediction = tf.equal(tf.argmax(task_output[test_task], 1), tf.argmax(y_[test_task], 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            test_x, test_y = task_permutation[test_task][2], task_permutation[test_task][3]
            test_img, test_lab = shuffle(len(test_x), test_x, test_y)
            iters = len(test_x) // minibatch_size
            te_acc = []
            for k in range(0, iters):
                test_batch_x = test_img[k * minibatch_size:(k + 1) * minibatch_size]
                test_batch_y = test_lab[k * minibatch_size:(k + 1) * minibatch_size]
                test_batch_images = read_images(test_batch_x)
                test_batch_labels = onehot(test_batch_y, out_dim[test_task])
                acc_ = sess.run(accuracy, feed_dict={x: test_batch_images, y_[test_task]: test_batch_labels})
                te_acc.append(acc_)
            test_avg = np.mean(te_acc)
            avg_accuracy += test_avg
            if test_task == 0:
                first_performance.append(test_avg)
            if test_task == task:
                last_performance.append(test_avg)
            tasks_acc.append(test_avg)
            print("Task: ",test_task," \tAccuracy: ",test_avg)
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
file= open('log-lwf.txt', 'w')
for fp in best_acc:
    file.write(str(fp))
    file.write('\n')

file.write(str(per_avg_performance))
file.close()
sess.close()

