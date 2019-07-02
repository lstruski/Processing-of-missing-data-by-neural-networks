import os
from datetime import datetime
from time import time

import numpy as np
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

# Training Parameters
learning_rate = 0.01
n_epochs = 250
batch_size = 64

# Network Parameters
num_hidden_1 = 256  # 1st layer num features
num_hidden_2 = 128  # 2nd layer num features (the latent dim)
num_hidden_3 = 64  # 3nd layer num features (the latent dim)
num_input = 784  # MNIST data_rbfn input (img shape: 28*28)

n_distribution = 5  # number of n_distribution

width_mask = 13  # size of window mask

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])

initializer = tf.contrib.layers.variance_scaling_initializer()

weights = {
    'encoder_h1': tf.Variable(initializer([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(initializer([num_hidden_1, num_hidden_2])),
    'encoder_h3': tf.Variable(initializer([num_hidden_2, num_hidden_3])),
    'decoder_h1': tf.Variable(initializer([num_hidden_3, num_hidden_2])),
    'decoder_h2': tf.Variable(initializer([num_hidden_2, num_hidden_1])),
    'decoder_h3': tf.Variable(initializer([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([num_hidden_3])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b2': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b3': tf.Variable(tf.random_normal([num_input])),
}


def random_mask(width_window, margin=0):
    margin_left = margin
    margin_righ = margin
    margin_top = margin
    margin_bottom = margin
    start_width = margin_top + np.random.randint(28 - width_window - margin_top - margin_bottom)
    start_height = margin_left + np.random.randint(28 - width_window - margin_left - margin_righ)

    return np.concatenate([28 * i + np.arange(start_height, start_height + width_window) for i in
                           np.arange(start_width, start_width + width_window)], axis=0).astype(np.int32)


def data_with_mask(x, width_window=10):
    h = width_window
    for i in range(x.shape[0]):
        if width_window <= 0:
            h = np.random.randint(8, 20)
        maska = random_mask(h)
        x[i, maska] = np.nan
    return x


def nr(mu, sigma):
    non_zero = tf.not_equal(sigma, 0.)
    new_sigma = tf.where(non_zero, sigma, tf.fill(tf.shape(sigma), 1e-20))
    sqrt_sigma = tf.sqrt(new_sigma)

    w = tf.div(mu, sqrt_sigma)
    nr_values = sqrt_sigma * (tf.div(tf.exp(tf.div(-tf.square(w), 2.)), np.sqrt(2 * np.pi)) +
                              tf.multiply(tf.div(w, 2.), 1 + tf.erf(tf.div(w, np.sqrt(2)))))

    nr_values = tf.where(non_zero, nr_values, (mu + tf.abs(mu)) / 2.)
    return nr_values


def conv_first(x, means, covs, p, gamma):
    gamma_ = tf.abs(gamma)
    # gamma_ = tf.cond(tf.less(gamma_[0], 1.), lambda: gamma_, lambda: tf.square(gamma_))
    covs_ = tf.abs(covs)
    p_ = tf.nn.softmax(p, axis=0)

    check_isnan = tf.is_nan(x)
    check_isnan = tf.reduce_sum(tf.cast(check_isnan, tf.int32), 1)

    x_miss = tf.gather(x, tf.reshape(tf.where(check_isnan > 0), [-1]))  # data_rbfn with missing values
    x = tf.gather(x, tf.reshape(tf.where(tf.equal(check_isnan, 0)), [-1]))  # data_rbfn without missing values

    # data_rbfn without missing
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))

    # data_rbfn with missing
    where_isnan = tf.is_nan(x_miss)
    where_isfinite = tf.is_finite(x_miss)
    size = tf.shape(x_miss)

    weights2 = tf.square(weights['encoder_h1'])

    # Collect distributions
    distributions = tf.TensorArray(dtype=x.dtype, size=n_distribution)
    q_collector = tf.TensorArray(dtype=x.dtype, size=n_distribution)

    # Each loop iteration calculates all per component
    def calculate_component(i, collect1, collect2):
        data_miss = tf.where(where_isnan, tf.reshape(tf.tile(means[i, :], [size[0]]), [-1, size[1]]), x_miss)
        miss_cov = tf.where(where_isnan, tf.reshape(tf.tile(covs_[i, :], [size[0]]), [-1, size[1]]),
                            tf.zeros([size[0], size[1]]))

        layer_1_m = tf.add(tf.matmul(data_miss, weights['encoder_h1']), biases['encoder_b1'])
        layer_1_m = nr(layer_1_m, tf.matmul(miss_cov, weights2))

        norm = tf.subtract(data_miss, means[i, :])
        norm = tf.square(norm)
        q = tf.where(where_isfinite,
                     tf.reshape(tf.tile(tf.add(gamma_, covs_[i, :]), [size[0]]), [-1, size[1]]),
                     tf.ones_like(x_miss))
        norm = tf.div(norm, q)
        norm = tf.reduce_sum(norm, axis=1)

        q = tf.log(q)
        q = tf.reduce_sum(q, axis=1)

        q = tf.add(q, norm)

        norm = tf.cast(tf.reduce_sum(tf.cast(where_isfinite, tf.int32), axis=1), tf.float32)
        norm = tf.multiply(norm, tf.log(2 * np.pi))

        q = tf.add(q, norm)
        q = -0.5 * q

        return i + 1, collect1.write(i, layer_1_m), collect2.write(i, q)

    i = tf.constant(0)
    _, final_distributions, final_q = tf.while_loop(lambda i, c1, c2: i < n_distribution, calculate_component,
                                                    loop_vars=(i, distributions, q_collector),
                                                    swap_memory=True, parallel_iterations=1)

    distrib = final_distributions.stack()
    log_q = final_q.stack()

    log_q = tf.add(log_q, tf.log(p_))
    r = tf.nn.softmax(log_q, axis=0)

    layer_1_miss = tf.multiply(distrib, r[:, :, tf.newaxis])
    layer_1_miss = tf.reduce_sum(layer_1_miss, axis=0)

    # join layer for data_rbfn with missing values with layer for data_rbfn without missing values
    layer_1 = tf.concat((layer_1, layer_1_miss), axis=0)
    return layer_1


# Building the encoder
def encoder(x, means, covs, p, gamma):
    layer_1 = conv_first(x, means, covs, p, gamma)

    # Encoder Hidden layer with sigmoid activation
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))
    return layer_3


# Building the decoder
def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']), biases['decoder_b3']))
    return layer_3


def prep_x(x):
    check_isnan = tf.is_nan(x)
    check_isnan = tf.reduce_sum(tf.cast(check_isnan, tf.int32), 1)

    x_miss = tf.gather(x, tf.reshape(tf.where(check_isnan > 0), [-1]))
    x = tf.gather(x, tf.reshape(tf.where(tf.equal(check_isnan, 0)), [-1]))
    return tf.concat((x, x_miss), axis=0)


t0 = time()
mnist = tf.keras.datasets.mnist
try:
    with np.load('./data/mnist.npz') as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
except FileNotFoundError:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print("Read data done in %0.3fs." % (time() - t0))

data_train = x_train

# choose test images nn * 10
nn = 100
data_test = x_test[np.where(y_test == 0)[0][:nn], :]
for i in range(1, 10):
    data_test = np.concatenate([data_test, x_test[np.where(y_test == i)[0][:nn], :]], axis=0)
data_test = np.random.permutation(data_test)

del mnist

# change background to white
data_train = 1. - data_train.reshape(-1, num_input)
data_test = 1. - data_test.reshape(-1, num_input)

# create missing data_rbfn train
data_train = data_with_mask(data_train, width_mask)

# create missing data_rbfn test
data_test = data_with_mask(data_test, width_mask)

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
data = imp.fit_transform(data_train)

t0 = time()
gmm = GaussianMixture(n_components=n_distribution, covariance_type='diag').fit(data)
print("GMM done in %0.3fs." % (time() - t0))

p = tf.Variable(initial_value=np.log(gmm.weights_.reshape((-1, 1))), dtype=tf.float32)
means = tf.Variable(initial_value=gmm.means_, dtype=tf.float32)
covs = tf.Variable(initial_value=gmm.covariances_, dtype=tf.float32)
gamma = tf.Variable(initial_value=tf.random_normal(shape=(1,), mean=1., stddev=1.), dtype=tf.float32)
del data, gmm

# Construct model
encoder_op = encoder(X, means, covs, p, gamma)
decoder_op = decoder(encoder_op)

y_pred = decoder_op  # prediction
y_true = prep_x(X)  # Targets (Labels) are the input data_rbfn.

where_isnan = tf.is_nan(y_true)
y_pred = tf.where(where_isnan, tf.zeros_like(y_pred), y_pred)
y_true = tf.where(where_isnan, tf.zeros_like(y_true), y_true)

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

trn_summary = [[] for _ in range(5)]
trn_imgs = [[] for _ in range(2)]
with tf.name_scope('train'):
    trn_summary[0] = tf.summary.scalar('loss', loss)
    trn_summary[1] = tf.summary.histogram("p", tf.nn.softmax(p, axis=0))
    for i in range(n_distribution):
        trn_summary[2].append(tf.summary.histogram("mean/{:d}".format(i), means[i]))
        trn_summary[3].append(tf.summary.histogram("cov/{:d}".format(i), tf.abs(covs[i])))
    trn_summary[4] = tf.summary.scalar('gamma', tf.abs(gamma)[0])
    image_grid = tf.contrib.gan.eval.image_grid(tf.gather(prep_x(X), np.arange(25)), (5, 5), (28, 28), 1)
    trn_imgs[0] = tf.summary.image('input', image_grid, 1)
    image_grid = tf.contrib.gan.eval.image_grid(tf.gather(decoder_op, np.arange(25)), (5, 5), (28, 28), 1)
    trn_imgs[1] = tf.summary.image('output', image_grid, 1)

tst_summary = [[] for _ in range(3)]
with tf.name_scope('test'):
    tst_summary[0] = tf.summary.scalar('loss', loss)
    image_grid = tf.contrib.gan.eval.image_grid(tf.gather(prep_x(X), np.arange(25)), (5, 5), (28, 28), 1)
    tst_summary[1] = tf.summary.image('input', image_grid, 1)
    image_grid = tf.contrib.gan.eval.image_grid(tf.gather(decoder_op, np.arange(25)), (5, 5), (28, 28), 1)
    tst_summary[2] = tf.summary.image('output', image_grid, 1)

current_date = datetime.now()
current_date = current_date.strftime('%d%b_%H%M%S')

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('./log/{}'.format(current_date), sess.graph)
    sess.run(init)  # run the initializer

    res = sess.run([*trn_summary], feed_dict={X: data_test[:25]})

    train_writer.add_summary(res[1], -1)
    for i in range(n_distribution):
        train_writer.add_summary(res[2][i], -1)
        train_writer.add_summary(res[3][i], -1)
    train_writer.add_summary(res[4], -1)

    epoch_tqdm = tqdm(range(1, n_epochs + 1), desc="Loss", leave=False)
    for epoch in epoch_tqdm:
        n_batch = data_train.shape[0] // batch_size
        for iteration in tqdm(range(n_batch), desc="Batches", leave=False):
            batch_x = data_train[(iteration * batch_size):((iteration + 1) * batch_size), :]

            # Run optimization op (backprop) and cost op (to get loss value)
            res = sess.run([optimizer, loss, *trn_summary, *trn_imgs], feed_dict={X: batch_x})

            train_writer.add_summary(res[-2], n_batch * (epoch - 1) + iteration)
            train_writer.add_summary(res[-1], n_batch * (epoch - 1) + iteration)
            train_writer.add_summary(res[2], n_batch * (epoch - 1) + iteration)
            train_writer.add_summary(res[3], n_batch * (epoch - 1) + iteration)
            for i in range(n_distribution):
                train_writer.add_summary(res[4][i], n_batch * (epoch - 1) + iteration)
                train_writer.add_summary(res[5][i], n_batch * (epoch - 1) + iteration)
            train_writer.add_summary(res[6], n_batch * (epoch - 1) + iteration)

            epoch_tqdm.set_description("Loss: {:.5f}".format(res[1]))

        tst_loss, tst_input, tst_output = sess.run([*tst_summary], feed_dict={X: data_test[:25]})
        train_writer.add_summary(tst_loss, epoch)
        train_writer.add_summary(tst_input, epoch)
        train_writer.add_summary(tst_output, epoch)
