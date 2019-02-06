#!/usr/bin/env python

import argparse
import os
import sys
from time import time

import numpy as np
import tensorflow as tf
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelBinarizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


# read data potentially with missing values
def read_data(path, sep=',', val_type='f8'):
    return np.genfromtxt(path, dtype=val_type, delimiter=sep)


def scaler_range(X, feature_range=(-1, 1), min_x=None, max_x=None):
    if min_x is None:
        min_x = np.nanmin(X, axis=0)
        max_x = np.nanmax(X, axis=0)

    X_std = (X - min_x) / (max_x - min_x)
    X_scaled = X_std * (feature_range[1] - feature_range[0]) + feature_range[0]
    return X_scaled, min_x, max_x


def nr(w):
    return tf.div(tf.exp(tf.div(-tf.pow(w, 2), 2.)), np.sqrt(2 * np.pi)) + tf.multiply(tf.div(w, 2.), 1 + tf.erf(
        tf.div(w, np.sqrt(2))))


# Create model
def multilayer_perceptron(x, means, covs, p, gamma, n_distribution, weights, biases, num_hidden_1):
    gamma = tf.abs(gamma)
    gamma = tf.cond(tf.less(gamma[0], 1.), lambda: gamma, lambda: tf.pow(gamma, 2))
    covs = tf.abs(covs)
    p = tf.abs(p)
    p = tf.div(p, tf.reduce_sum(p, axis=0))

    check_isnan = tf.is_nan(x)
    check_isnan = tf.reduce_sum(tf.cast(check_isnan, tf.int32), 1)

    x_miss = tf.gather(x, tf.reshape(tf.where(check_isnan > 0), [-1]))  # data with missing values
    x = tf.gather(x, tf.reshape(tf.where(tf.equal(check_isnan, 0)), [-1]))  # data without missing values

    # data without missing
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))

    # data with missing
    where_isnan = tf.is_nan(x_miss)
    where_isfinite = tf.is_finite(x_miss)
    size = tf.shape(x_miss)

    weights2 = tf.square(weights['h1'])

    Q = []
    layer_1_miss = tf.zeros([size[0], num_hidden_1])
    for i in range(n_distribution):
        data_miss = tf.where(where_isnan, tf.reshape(tf.tile(means[i, :], [size[0]]), [-1, size[1]]), x_miss)
        miss_cov = tf.where(where_isnan, tf.reshape(tf.tile(covs[i, :], [size[0]]), [-1, size[1]]),
                            tf.zeros([size[0], size[1]]))

        layer_1_m = tf.add(tf.matmul(data_miss, weights['h1']), biases['b1'])

        layer_1_m = tf.div(layer_1_m, tf.sqrt(tf.matmul(miss_cov, weights2)))
        layer_1_m = nr(layer_1_m)

        layer_1_miss = tf.cond(tf.equal(tf.constant(i), tf.constant(0)), lambda: tf.add(layer_1_miss, layer_1_m),
                               lambda: tf.concat((layer_1_miss, layer_1_m), axis=0))

        # calculate q_i
        norm = tf.subtract(data_miss, means[i, :])
        norm = tf.square(norm)
        q = tf.where(where_isfinite,
                     tf.reshape(tf.tile(tf.add(gamma, covs[i, :]), [size[0]]), [-1, size[1]]),
                     tf.ones_like(x_miss))
        norm = tf.div(norm, q)

        norm = tf.reduce_sum(norm, axis=1)

        q = tf.log(q)
        q = tf.reduce_sum(q, axis=1)

        q = tf.add(q, norm)

        norm = tf.cast(tf.reduce_sum(tf.cast(where_isfinite, tf.int32), axis=1), tf.float32)
        norm = tf.multiply(norm, tf.log(2 * np.pi))

        q = tf.add(q, norm)
        q = tf.multiply(q, -0.5)

        Q = tf.concat((Q, q), axis=0)

    Q = tf.reshape(Q, shape=(n_distribution, -1))
    Q = tf.add(Q, tf.log(p))
    Q = tf.subtract(Q, tf.reduce_max(Q, axis=0))
    Q = tf.where(Q < -20, tf.multiply(tf.ones_like(Q), -20), Q)
    Q = tf.exp(Q)
    Q = tf.div(Q, tf.reduce_sum(Q, axis=0))
    Q = tf.reshape(Q, shape=(-1, 1))

    layer_1_miss = tf.multiply(layer_1_miss, Q)
    layer_1_miss = tf.reshape(layer_1_miss, shape=(n_distribution, size[0], num_hidden_1))
    layer_1_miss = tf.reduce_sum(layer_1_miss, axis=0)

    # join layer for data with missing values with layer for data without missing values
    layer_1 = tf.concat((layer_1, layer_1_miss), axis=0)

    # Encoder Hidden layer with sigmoid activation
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['h4']), biases['b4']))
    return tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])


def prep_labels(x, y):
    check_isnan = tf.is_nan(x)
    check_isnan = tf.reduce_sum(tf.cast(check_isnan, tf.int32), 1)

    y_miss = tf.gather(y, tf.reshape(tf.where(check_isnan > 0), [-1]))
    y = tf.gather(y, tf.reshape(tf.where(tf.equal(check_isnan, 0)), [-1]))
    return tf.concat((y, y_miss), axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./', help='Directory for input data')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
    parser.add_argument('--training_epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--n_distribution', type=int, default=3, help='Number of distributions')
    FLAGS, unparsed = parser.parse_known_args()
    # print([sys.argv[0]] + unparsed)
    path_dir = FLAGS.data_dir

    # Parameters
    learning_rate = FLAGS.learning_rate
    batch_size = FLAGS.batch_size
    training_epochs = FLAGS.training_epochs

    # Network Parameters
    n_distribution = FLAGS.n_distribution

    data = read_data(os.path.join(path_dir, 'data.txt'))
    data, minx, maxx = scaler_range(data, feature_range=(-1, 1))

    labels = read_data(os.path.join(path_dir, 'labels.txt'))
    lb = LabelBinarizer()
    lb.fit(labels)

    class_name = lb.classes_
    n_class = class_name.shape[0]
    if n_class == 2:
        lb.fit(np.append(labels, np.max(class_name) + 1))

    n_features = data.shape[1]
    num_hidden_1 = int(0.5 * n_features)
    num_hidden_2 = num_hidden_1
    num_hidden_3 = num_hidden_1
    num_hidden_4 = num_hidden_1
    num_hidden_5 = n_class

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')

    complate_data = imp.fit_transform(data)
    gmm = GaussianMixture(n_components=n_distribution, covariance_type='diag').fit(complate_data)
    del complate_data, imp

    gmm_weights = gmm.weights_.reshape((-1, 1))
    gmm_means = gmm.means_
    gmm_covariances = gmm.covariances_
    del gmm

    acc = np.zeros((3, 5))

    time_train = np.zeros(5)
    time_test = np.zeros(5)

    skf = StratifiedKFold(n_splits=5)
    id_acc = 0
    for trn_index, test_index in skf.split(data, labels):
        X_train = data[trn_index]
        X_lab = labels[trn_index]
        train_index, valid_index = next(StratifiedKFold(n_splits=5).split(X_train, X_lab))

        train_x = X_train[train_index, :]
        valid_x = X_train[valid_index, :]
        test_x = data[test_index, :]

        train_y = lb.transform(X_lab[train_index])
        valid_y = lb.transform(X_lab[valid_index])
        test_y = lb.transform(labels[test_index])
        if n_class == 2:
            train_y = train_y[:, :-1]
            valid_y = valid_y[:, :-1]
            test_y = test_y[:, :-1]

        with tf.Graph().as_default() as graph:

            initializer = tf.contrib.layers.variance_scaling_initializer()

            weights = {
                'h1': tf.Variable(initializer([n_features, num_hidden_1])),
                'h2': tf.Variable(initializer([num_hidden_1, num_hidden_2])),
                'h3': tf.Variable(initializer([num_hidden_2, num_hidden_3])),
                'h4': tf.Variable(initializer([num_hidden_3, num_hidden_4])),
                'h5': tf.Variable(initializer([num_hidden_4, num_hidden_5])),
            }
            biases = {
                'b1': tf.Variable(tf.random_normal([num_hidden_1])),
                'b2': tf.Variable(tf.random_normal([num_hidden_2])),
                'b3': tf.Variable(tf.random_normal([num_hidden_3])),
                'b4': tf.Variable(tf.random_normal([num_hidden_4])),
                'b5': tf.Variable(tf.random_normal([num_hidden_5])),
            }

            # Symbols
            z = tf.placeholder(shape=[None, n_features], dtype=tf.float32)
            y = tf.placeholder(shape=[None, n_class], dtype=tf.float32)

            p = tf.Variable(initial_value=gmm_weights, dtype=tf.float32)
            means = tf.Variable(initial_value=gmm_means, dtype=tf.float32)
            covs = tf.Variable(initial_value=gmm_covariances, dtype=tf.float32)

            gamma = tf.Variable(initial_value=tf.random_normal(shape=(1,), mean=2, stddev=1.), dtype=tf.float32)

            # Construct model
            predict = multilayer_perceptron(z, means, covs, p, gamma, n_distribution, weights, biases, num_hidden_1)

            y_true = prep_labels(z, y)

            # Mean squared error
            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predict, labels=y_true))

            l_r = learning_rate
            # Gradient descent
            optimizer = tf.train.GradientDescentOptimizer(l_r).minimize(cost)

            # Initialize the variables (i.e. assign their default value)
            init = tf.global_variables_initializer()

            nr_epoch = 10

            val_weights = None
            val_biases = None
            val_p = None
            val_means = None
            val_covs = None
            val_gamma = None

            with tf.Session(graph=graph) as sess:
                sess.run(init)

                min_cost = np.inf
                n_cost_up = 0

                prev_train_cost = np.inf

                time_train[id_acc] = time()

                epoch = 0
                # Training cycle
                for epoch in range(training_epochs):
                    # print("\r[{}|{}] Step: {:d} from 5".format(epoch + 1, training_epochs, id_acc), end="")
                    # sys.stdout.flush()

                    curr_train_cost = []
                    for batch_idx in range(0, train_y.shape[0], batch_size):
                        x_batch = train_x[batch_idx:batch_idx + batch_size, :]
                        y_batch = train_y[batch_idx:batch_idx + batch_size, :]

                        temp_train_cost, _ = sess.run([cost, optimizer], feed_dict={z: x_batch, y: y_batch})
                        curr_train_cost.append(temp_train_cost)

                    curr_train_cost = np.asarray(curr_train_cost).mean()

                    if epoch > nr_epoch and (prev_train_cost - curr_train_cost) < 1e-4 < l_r:
                        l_r = l_r / 2.
                        optimizer = tf.train.GradientDescentOptimizer(l_r).minimize(cost)

                    prev_train_cost = curr_train_cost

                    curr_cost = []
                    for batch_idx in range(0, valid_y.shape[0], batch_size):
                        x_batch = valid_x[batch_idx:batch_idx + batch_size, :]
                        y_batch = valid_y[batch_idx:batch_idx + batch_size, :]
                        curr_cost.append(sess.run(cost, feed_dict={z: x_batch, y: y_batch}))

                    curr_cost = np.asarray(curr_cost).mean()

                    if min_cost > curr_cost:
                        min_cost = curr_cost
                        n_cost_up = 0

                        val_weights = {
                            'h1': weights['h1'].eval(),
                            'h2': weights['h2'].eval(),
                            'h3': weights['h3'].eval(),
                            'h4': weights['h4'].eval(),
                            'h5': weights['h5'].eval(),
                        }
                        val_biases = {
                            'b1': biases['b1'].eval(),
                            'b2': biases['b2'].eval(),
                            'b3': biases['b3'].eval(),
                            'b4': biases['b4'].eval(),
                            'b5': biases['b5'].eval(),
                        }

                        val_p = p.eval()
                        val_means = means.eval()
                        val_covs = covs.eval()
                        val_gamma = gamma.eval()
                    elif epoch > nr_epoch:
                        n_cost_up = n_cost_up + 1

                    if n_cost_up == 5 and 1e-4 < l_r:
                        l_r = l_r / 2.
                        optimizer = tf.train.GradientDescentOptimizer(l_r).minimize(cost)
                    elif n_cost_up == 10:
                        break
                        
                time_train[id_acc] = (time() - time_train[id_acc]) / (epoch + 1)

                predict = multilayer_perceptron(z, val_means, val_covs, val_p, val_gamma, n_distribution,
                                                val_weights, val_biases, num_hidden_1)

                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(predict, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                train_accuracy = []
                for batch_idx in range(0, train_y.shape[0], batch_size):
                    x_batch = train_x[batch_idx:batch_idx + batch_size, :]
                    y_batch = train_y[batch_idx:batch_idx + batch_size, :]

                    train_accuracy.append(accuracy.eval({z: x_batch, y: y_batch}))
                train_accuracy = np.mean(train_accuracy)

                valid_accuracy = []
                for batch_idx in range(0, valid_y.shape[0], batch_size):
                    x_batch = valid_x[batch_idx:batch_idx + batch_size, :]
                    y_batch = valid_y[batch_idx:batch_idx + batch_size, :]

                    valid_accuracy.append(accuracy.eval({z: x_batch, y: y_batch}))
                valid_accuracy = np.mean(valid_accuracy)

                time_test[id_acc] = time()
                test_accuracy = []
                for batch_idx in range(0, test_y.shape[0], batch_size):
                    x_batch = test_x[batch_idx:batch_idx + batch_size, :]
                    y_batch = test_y[batch_idx:batch_idx + batch_size, :]
                    test_accuracy.append(accuracy.eval({z: x_batch, y: y_batch}))
                test_accuracy = np.mean(test_accuracy)
                time_test[id_acc] = time() - time_test[id_acc]

                acc[0, id_acc] = train_accuracy
                acc[1, id_acc] = valid_accuracy
                acc[2, id_acc] = test_accuracy
                id_acc = id_acc + 1

    mean_acc = np.average(acc, axis=1)
    std_acc = np.std(acc, axis=1)
    sys.stdout.flush()

    print(
        "{:.4f};{:.4f};{:.4f};{:.4f};{:.4f};{:.4f};{:.4f};{:.4f};{};{};{};{}".format(
            mean_acc[0], std_acc[0], mean_acc[1], std_acc[1], mean_acc[2], std_acc[2], np.average(time_train),
            np.average(time_test), FLAGS.learning_rate, FLAGS.batch_size, FLAGS.training_epochs, FLAGS.n_distribution))


if __name__ == '__main__':
    main()
