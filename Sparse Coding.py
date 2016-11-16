# Might just use Scit-Kit Learn's implementation
# Wont work probably

import numpy as np
import tensorflow as tf

def getData():
    return np.random.random((200,50))

def main():
    x = tf.placeholder(tf.float32, shape = [200, 50])
    xShape = x.get_shape().as_list()
    k = 40
    h = tf.Variable(tf.random_normal(shape = [xShape[0], k]))
    D = tf.Variable(tf.random_normal(shape = [k, xShape[1]]))
    cost = tf.reduce_sum(tf.square(x - tf.reduce_sum(tf.matmul(h, D)))) + tf.reduce_sum(tf.log(1 + tf.square(h)))
    train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(x, tf.matmul(h, D)), tf.float32))

    with tf.Session() as sess :
        sess.run(tf.initialize_all_variables())
        Xtrain = getData()
        for i in range(1000):
            sess.run(train, feed_dict = {x : Xtrain})
        print(Xtrain)
        print(sess.run(h))
        print(sess.run(D))

if __name__ == '__main__':
    main()