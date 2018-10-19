#!/usr/bin/env python3



# TODO finish this advanced script



import tensorflow as tf
# MNIST hand-written digits dataset
mnist = tf.keras.datasets.mnist


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

class Model:
    def __init__(self, epochs=5, batch_sz=16):
        """Constructor"""

        # Input data Tensor
        x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1]) 
        # Training labels
        y = tf.placeholder(tf.float32, shape=[None, 10]) 

        # Model
        conv1 = tf.layers.conv2d(x, 8, 3,
                activation=tf.nn.relu, padding='same')
        conv2 = tf.layers.conv2d(conv1, 16, 3, 
                activation=tf.nn.relu, padding='same')
        conv3 = tf.layers.conv2d(conv2, 32, 3, 
                activation=tf.nn.relu, padding='same')
            
        flat = tf.layers.flatten(conv3) # flatten the data to vectors

        fc4 = tf.layers.dense(flat, units=256,
                activation=tf.nn.relu)
        fc5 = tf.layers.dense(fc4, units=128,
                activation=tf.nn.relu)
        self.logits = tf.layers.dense(fc5, units=10)
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=self.logits)

    def to_tfrecord(self, x_train, y_train):
        """Write data to tfrecord file"""

        with tf.python_io.TFRecordWriter("train.tfrecord") as writer:
            for i in range(len(x_train)):
                print("Working on example %d" % i)

                # Load the image
                img = x_train[i].flatten()
                label = y_train[i].flatten()

                # Create a feature
                feature = {'label': _floats_feature(label),
                           'image': _floats_feature(img)}
                # Create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                
                # Serialize to string and write on the file
                writer.write(example.SerializeToString())



    def train(self, x_train, y_train):
        """Train the model"""
        
        # Transform digits to one-hot encoded vectors
        y_train = tf.one_hot(y_train, 10)
        y_test = tf.one_hot(y_test, 10)


if __name__ == '__main__':

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    # Scale data  to [0,1]
    x_train, x_test = x_train / 255.0, x_test / 255.0

    print("Training data shape:", x_train.shape)
    print("Testing data shape:", x_test.shape)
    print("Training label shape:", y_train.shape)
    print("Testing label shape:", y_test.shape)

    model = Model()
    model.to_tfrecord(x_train, y_train)




