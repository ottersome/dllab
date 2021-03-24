from tensorflow.examples.tutorials.mnist import input_data
from __future__import absolute_import, division, print_function
import tensrflow as tf
from tensorflow.keras import Model,layers
import numpy as np

mnist = input_data.read_data_sets("/tmp/data/"), one_hot=True)

X_train = mnist.train.images
Y_train = mnist.train.labels
X_test = mnist.test.images
Y_test = mnist.test.images
x_train,x_test = np.array(x_train,np.float32), np.array(x_test,np.float32)
X_train = X_train/255
X_test= X_test/255


numOfClasses = 10

learning_rate = 0.001
training_steps=200
batch_size = 128
display_step = 10

# Network Parametrs
#
conv1_filters = 32
conv2_filters = 64
fc1_units = 1024

train_data = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_data = trdef cross_entropy_loss(x, y):
    # Convert labels to int 64 for tf cross-entropy function.
    y = tf.cast(y, tf.int64)
    # Apply softmax to logits and compute cross-entropy.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    # Average loss across the batch.
    return tf.reduce_mean(loss)

# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)ain_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

class ConvolutionNetwork(Model):
    super(ConvolutionNetwork,self).__init__()
    self.conv1 = layers.Conv2D(32,kernel_size=5,activation=tf.nn.relu)
    self.maxpool1 = layers.MaxPool2D(2, strides=2)

    self.conv2 = layers.Conv2D(64,kernel_size=3,activation=tf.nn.relu)
    self.maxpool2  = layers.MaxPool2D(2,strides=2)

    self.flatten = layers.Flatten()

    # Now we start with the fully connected layers
    self.fc1 = layers.Dense(1024)
    self.dropout = layers.Dropout(rate=0.5)
    self.out = layers.Dense(numOfClasses)

    def call(self, x, is_training=False):
        x = tf.reshape(x,[-1,28,28,1])
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout()
        x = self.out(x)
        if not is_training:
            x = tf.nn.softmax(x)
        return x

convNet = ConvNet()
optimizer = tf.optimizers.Adam(learning_rate)

def run_opt



