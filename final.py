# Load pickled data
import pickle
import numpy as np
import keras

n_classes=43

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'traffic-signs-data/train.p'
validation_file='traffic-signs-data/valid.p'
testing_file = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


#####################################################################################

import tensorflow as tf

row = 32
col = 32
channel=3

X_train = X_train.reshape(X_train.shape[0],row,col,channel)
X_valid = X_valid.reshape(X_valid.shape[0],row,col,channel)
X_test = X_test.reshape(X_test.shape[0],row,col,channel)
i_shape = (row,col,channel)

X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_valid /= 255
X_test /= 255

 
y_train =  tf.keras.utils.to_categorical(y_train, n_classes)
y_valid = tf.keras.utils.to_categorical(y_valid, n_classes)
y_test =  tf.keras.utils.to_categorical(y_test, n_classes)
#y_train =  keras.utils.to_categorical(y_train, n_classes)
#y_valid =  keras.utils.to_categorical(y_valid, n_classes)
#y_test =   keras.utils.to_categorical(y_test, n_classes)
##############################################################3
 

import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# Input layer
x  = tf.placeholder(tf.float32, [None, 32*32*3], name='x')
y_ = tf.placeholder(tf.float32, [None, 43],  name='y_')
x_image = tf.reshape(x, [-1, 32, 32, 3])

# Convolutional layer 1
W_conv1 = weight_variable([3, 3, 3, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Convolutional layer 2
W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Fully connected layer 1
h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])

W_fc1 = weight_variable([8 * 8 * 64, 1024])
b_fc1 = bias_variable([1024])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob  = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Fully connected layer 2 (Output layer)
W_fc2 = weight_variable([1024, 43])
b_fc2 = bias_variable([43])

y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y')

# Evaluation functions
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

# Training algorithm
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Training steps
with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())

  max_steps = 100
  for step in range(max_steps):
    
    
    n100  = np.random.choice( len(X_train)-1, 100, replace=False)
    Xbatch = X_train[n100].reshape((-1,32*32*3))
    ybatch = y_train[n100]
    
    sess.run(train_step, feed_dict={x: Xbatch, y_:ybatch, keep_prob: 0.5})
    print("done...  ", step/ max_steps)
    
  print(max_steps, sess.run(accuracy, feed_dict={x: X_test.reshape((-1,32*32*3)), 
                                                 y_: y_test, keep_prob: 1.0}))
