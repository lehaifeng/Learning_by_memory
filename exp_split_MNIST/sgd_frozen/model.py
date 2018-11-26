import tensorflow as tf

class Model:
    def __init__(self, x, y1,y2,y3,y4,y5,keep_prob):
        self.x = x  # input placeholder
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3
        self.y4 = y4
        self.y5 = y5

        self.keep_prob=keep_prob
        # simple 3-layer network 784-512-256-10
        with tf.variable_scope("weight"):
            W1 = tf.get_variable('w1',shape=[784,64],initializer=tf.truncated_normal_initializer(stddev=0.1))
            b1 = tf.get_variable('b1',shape=[64],initializer=tf.constant_initializer(0.1))
            self.h1 = tf.nn.relu(tf.matmul(x, W1) + b1)  # hidden layer1

            W2 = tf.get_variable('w2',shape=[64,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
            b2 = tf.get_variable('b2', shape=[32], initializer=tf.constant_initializer(0.1))
            self.h2 = tf.nn.relu(tf.matmul(self.h1, W2) + b2)  # hidden layer1
            self.dropout1 = tf.nn.dropout(self.h2, self.keep_prob)

        with tf.variable_scope("output_1"):
            W3 = tf.get_variable('w3', shape=[32,2], initializer=tf.truncated_normal_initializer(stddev=0.1))
            b3 = tf.get_variable('b3', shape=[2], initializer=tf.constant_initializer(0.1))
            self.out1 = tf.matmul(self.dropout1, W3) + b3  # output layer
            self.probs1 = tf.nn.softmax(self.out1)
        with tf.variable_scope("output_2"):
            W3 = tf.get_variable('w3', shape=[32,2], initializer=tf.truncated_normal_initializer(stddev=0.1))
            b3 = tf.get_variable('b3', shape=[2], initializer=tf.constant_initializer(0.1))
            self.out2 = tf.matmul(self.dropout1, W3) + b3
            self.probs2 = tf.nn.softmax(self.out2)
        with tf.variable_scope("output_3"):
            W3 = tf.get_variable('w3', shape=[32,2], initializer=tf.truncated_normal_initializer(stddev=0.1))
            b3 = tf.get_variable('b3', shape=[2], initializer=tf.constant_initializer(0.1))
            self.out3 = tf.matmul(self.dropout1, W3) + b3
            self.probs3 = tf.nn.softmax(self.out3)
        with tf.variable_scope("output_4"):
            W3 = tf.get_variable('w3', shape=[32,2], initializer=tf.truncated_normal_initializer(stddev=0.1))
            b3 = tf.get_variable('b3', shape=[2], initializer=tf.constant_initializer(0.1))
            self.out4 = tf.matmul(self.dropout1, W3) + b3
            self.probs4 = tf.nn.softmax(self.out4)
        with tf.variable_scope("output_5"):
            W3 = tf.get_variable('w3', shape=[32,2], initializer=tf.truncated_normal_initializer(stddev=0.1))
            b3 = tf.get_variable('b3', shape=[2], initializer=tf.constant_initializer(0.1))
            self.out5 = tf.matmul(self.dropout1, W3) + b3
            self.probs5 = tf.nn.softmax(self.out5)
        


            self.var_list = [W1, b1, W2, b2, W3, b3]
            #self.var_list=tf.trainable_variables()

        # vanilla single-task loss
        with tf.variable_scope("loss"):
            self.cross_entropy1 = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y1, logits=self.out1))
            self.cross_entropy2 = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y2, logits=self.out2))
            self.cross_entropy3 = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y3, logits=self.out3))
            self.cross_entropy4 = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y4, logits=self.out4))
            self.cross_entropy5 = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y5, logits=self.out5))
            


        self.set_vanilla_loss1()
        self.set_vanilla_loss2()
        self.set_vanilla_loss3()
        self.set_vanilla_loss4()
        self.set_vanilla_loss5()
        

        # performance metrics
        with tf.variable_scope("accuracy"):
            correct_prediction1 = tf.equal(tf.argmax(self.out1, 1), tf.argmax(self.y1, 1))
            correct_prediction2 = tf.equal(tf.argmax(self.out2, 1), tf.argmax(self.y2, 1))
            correct_prediction3 = tf.equal(tf.argmax(self.out3, 1), tf.argmax(self.y3, 1))
            correct_prediction4 = tf.equal(tf.argmax(self.out4, 1), tf.argmax(self.y4, 1))
            correct_prediction5 = tf.equal(tf.argmax(self.out5, 1), tf.argmax(self.y5, 1))
            

            self.accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
            self.accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))
            self.accuracy3 = tf.reduce_mean(tf.cast(correct_prediction3, tf.float32))
            self.accuracy4 = tf.reduce_mean(tf.cast(correct_prediction4, tf.float32))
            self.accuracy5 = tf.reduce_mean(tf.cast(correct_prediction5, tf.float32))
            

    def star(self):
        # used for saving optimal weights after most recent task training
        self.star_vars = []

        for v in range(len(self.var_list)):
            self.star_vars.append(self.var_list[v].eval())


    def restore(self, sess):
        # reassign optimal weights for latest task
        if hasattr(self, "star_vars"):
            for v in range(len(self.var_list)):
                sess.run(self.var_list[v].assign(self.star_vars[v]))

    def set_vanilla_loss1(self):
        self.train_step1 = tf.train.GradientDescentOptimizer(0.001).minimize(self.cross_entropy1)
    def set_vanilla_loss2(self):
        # 选择待优化的参数
        output_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='output_2')
        #print(output_vars)
        self.train_step2 = tf.train.GradientDescentOptimizer(0.001).minimize(self.cross_entropy2,var_list=output_vars)
    def set_vanilla_loss3(self):
        output_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='output_3')
        self.train_step3 = tf.train.GradientDescentOptimizer(0.001).minimize(self.cross_entropy3,var_list=output_vars)
    def set_vanilla_loss4(self):
        output_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='output_4')
        self.train_step4 = tf.train.GradientDescentOptimizer(0.001).minimize(self.cross_entropy4,var_list=output_vars)
    def set_vanilla_loss5(self):
        output_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='output_5')
        self.train_step5 = tf.train.GradientDescentOptimizer(0.001).minimize(self.cross_entropy5,var_list=output_vars)
    