import tensorflow as tf

def weight_variable(shape,name='W'):
  return tf.get_variable(name=name, shape = shape, initializer = tf.contrib.layers.xavier_initializer())

def bias_variable(shape,name='bias'):
  initial = tf.constant(0.0, shape = shape)
  return tf.get_variable(name=name, initializer = initial)

# vgg9 layers without bn layer
class VGG:

    def __init__(self, x,training):
        with tf.device("/device:GPU:0"):
            conv1 = tf.layers.conv2d(x, filters=32, kernel_size=(3, 3), padding='SAME',
                                     use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
            relu1 = tf.nn.relu(conv1)
            dropout1 = tf.layers.dropout(relu1,training=training, rate=0.5)


            conv2 = tf.layers.conv2d(dropout1, filters=32, kernel_size=(3, 3), padding='SAME',
                                     use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
            relu2 = tf.nn.relu(conv2)

            pool2 = tf.layers.max_pooling2d(relu2, (2, 2), (2, 2), padding='SAME')

            conv3 = tf.layers.conv2d(pool2, filters=64, kernel_size=(3, 3), padding='SAME',
                                     use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
            relu3 = tf.nn.relu(conv3)
            dropout3 = tf.layers.dropout(relu3,training=training, rate=0.5)

            # Layer 4
            conv4 = tf.layers.conv2d(dropout3, filters=64, kernel_size=(3, 3), padding='SAME',
                                     use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
            relu4 = tf.nn.relu(conv4)
            pool4 = tf.layers.max_pooling2d(relu4, (2, 2), (2, 2), padding='SAME')

            # Layer 5
            conv5 = tf.layers.conv2d(pool4, filters=128, kernel_size=(3, 3), padding='SAME',
                                     use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
            relu5 = tf.nn.relu(conv5)
            dropout5 = tf.layers.dropout(relu5,training=training, rate=0.5)

            # Layer 6
            conv6 = tf.layers.conv2d(dropout5, filters=128, kernel_size=(3, 3), padding='SAME',
                                     use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
            relu6 = tf.nn.relu(conv6)
            pool6 = tf.layers.max_pooling2d(relu6, (2, 2), (2, 2), padding='SAME')

            flattened = tf.contrib.layers.flatten(pool6)
            f_dropout = tf.layers.dropout(flattened,training=training, rate=0.5)

            dense7 = tf.layers.dense(inputs=f_dropout, units=1024,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
            relu7 = tf.nn.relu(dense7)
            dropout7 = tf.layers.dropout(relu7, training=training, rate=0.5)

            dense8 = tf.layers.dense(inputs=dropout7, units=512,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
            relu8 = tf.nn.relu(dense8)
            dropout8 = tf.layers.dropout(relu8, training=training, rate=0.5)

            self.flatten = dropout8
            self.params = tf.trainable_variables()[0:16]

            return
