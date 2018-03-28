#################################################
#  This project creates logistic neural network #
#  implementation using Tensorflow.             #
#  Computes Xor function:                       #
#  - 4 hidden neoruns                           #
#  - 2 hidden neoruns                           #
#  - 1 hidden neorun                            #
#################################################

import tensorflow as tf
import numpy as nump


def xor_model(data_input, k, dim, nb_output, w1, w2, b1, b2):
    temp = 0.001
    expected_output = [[0], [1], [1], [0]]

    y = tf.placeholder(tf.float32, shape=[None, nb_output])
    x = tf.placeholder(tf.float32, shape=[None, dim])

    z1 = tf.matmul(x, w1) + b1
    h_layer1 = tf.sigmoid(z1 / temp)

    # 1 hidden neural
    if k == 1:
        h_layer1 = tf.concat([h_layer1, x], 1)

    z2 = tf.matmul(h_layer1, w2) + b2
    out = tf.sigmoid(z2 / temp)

    squared_deltas = tf.square(out - y)
    loss = tf.reduce_sum(squared_deltas)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)  # initialize variables

    out_put = sess.run([out, loss], {x: data_input, y: expected_output})
    return out_put


if __name__ == '__main__':
    x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]

    #####################################
    #   Xor preparation -4 hidden layer #
    #####################################
    dim = 2
    nb_outputs = 1
    nb_hidden = 4

    weight1 = tf.Variable(nump.array([-1., -1., 1., 1., -1., 1., -1., 1.]).reshape([dim, nb_hidden]), dtype=tf.float32)
    weight2 = tf.Variable(nump.array([-1., 1., 1., -1.]).reshape([nb_hidden, nb_outputs]), dtype=tf.float32)
    b1 = tf.Variable([-.5, -.5, -.5, -2.5], dtype=tf.float32)
    b2 = tf.Variable([-.5], dtype=tf.float32)

    output_xor = xor_model(x_train, nb_hidden, dim, nb_outputs, weight1, weight2, b1, b2)
    print("Xor function -", nb_hidden, "hidden layer result:\n", str(output_xor[0]), "\nloss is:\n", str(output_xor[1]))

    #####################################
    #   Xor preparation -3 hidden layer #
    #####################################
    dim = 2
    nb_hidden = 2
    nb_outputs = 1

    weight1 = tf.Variable([[1., -1.], [1., -1]])
    weight2 = tf.Variable([[1.], [1.]])
    b1 = tf.Variable([-.5, 1.5], tf.float32)
    b2 = tf.Variable([-1.5], tf.float32)

    output_xor = xor_model(x_train, nb_hidden, dim, nb_outputs, weight1, weight2, b1, b2)
    print("Xor function -", nb_hidden, "hidden layer result:\n", str(output_xor[0]), "\nloss is:\n", str(output_xor[1]))

    #####################################
    #   Xor preparation -1 hidden layer #
    #####################################
    dim = 2
    nb_hidden = 1
    nb_outputs = 1

    weight1 = tf.Variable([[1.], [1.]])
    weight2 = tf.Variable([[-2.], [1.], [1.]])
    b1 = tf.Variable([-1.5], tf.float32)
    b2 = tf.Variable([-0.5], tf.float32)
    output_xor = xor_model(x_train, nb_hidden, dim, nb_outputs, weight1, weight2, b1, b2)
    print("Xor function -", nb_hidden, "hidden layer result:\n", str(output_xor[0]), "\nloss is:\n", str(output_xor[1]))