#!/usr/bin/env python3

# See https://arxiv.org/abs/1606.05908

import tensorflow as tf
import numpy as np

# MNIST hand-written digits dataset
mnist = tf.keras.datasets.mnist

class VAE:
    def __init__(self, epochs=5, batch_sz=16):
        """Constructor"""

        # Input data Tensor
        x = tf.placeholder(tf.float32, shape=[None, 28**2]) 
        self.x = x
        fc1 = tf.layers.dense(x, units=256,
                activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, units=128,
                activation=tf.nn.relu)
        fc3 = tf.layers.dense(fc2, units=64,
                activation=tf.nn.relu)
        fc4 = tf.layers.dense(fc3, units=4,
                activation=None)
        
        mu = fc4[:,:2]

        # estimate log sigma^2 for numerical stability
        log_sig_sq = fc4[:,2:]

        # z = mu + sigma * epsilon
        # epsilon is a sample from a N(0, 1) distribution
        eps = tf.random_normal(tf.shape(mu), 0.0, 1.0, dtype=tf.float32)

        # Random normal variable for decoder :D
        z = tf.add(mu, tf.sqrt(tf.exp(log_sig_sq)) * eps, name='z')
        self.z = z

        fc5 = tf.layers.dense(z, units=64,
                activation=tf.nn.relu)
        fc6 = tf.layers.dense(fc5, units=128,
                activation=tf.nn.relu)
        fc7 = tf.layers.dense(fc6, units=256,
                activation=tf.nn.relu)
        
        # sigmoid bounds answer to [0,1] for images
        rec = tf.layers.dense(fc7, units=28**2,
                activation=tf.sigmoid)
        self.rec = rec
        # reconstruction loss
        self.rec_loss = tf.reduce_mean(
                      -tf.reduce_sum(x * tf.log(tf.clip_by_value(rec, 1e-10, 1.0))
                      + (1.0 - x) * tf.log(tf.clip_by_value(1.0 - rec, 1e-10, 1.0)),
                      axis=1))

        # stdev is the diagonal of the covariance matrix
        # KLD( (mu, sigma) , N(0,1) ) = .5 (tr(sigma2) + mu^T mu - k - log det(sigma2))

        self.kld = tf.reduce_mean(
                   -0.5 * (tf.reduce_sum(1.0 + log_sig_sq - tf.square(mu)
                   - tf.exp(log_sig_sq), axis=1)))

        self.loss = self.kld + self.rec_loss
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)

    def step(self, batch, sess):
        kld, rec_loss, _ = sess.run([self.kld, self.rec_loss, self.opt],
                            feed_dict={self.x: batch})
        return kld, rec_loss

if __name__ == '__main__':
    
    # ignore testing data and labels
    (x_train, _),_ = mnist.load_data()
    # Scale data  to [0,1]
    x_train = (x_train / 255.0).reshape(-1, 28**2)

    model = VAE()
    
    batch = 32
    iterations = 100000

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(iterations):
            batch_idx = np.random.randint(len(x_train), size=batch)
            x_batch = x_train[batch_idx, :]
    
            kld, rec_loss = model.step(x_batch, sess)

            if i % 100 == 0:
                print("Iteration %d: KLD = %f, RecLoss = %f" % (i, kld, rec_loss))


