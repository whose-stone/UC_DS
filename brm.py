import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class RBM (object): 
  def __init__(self, input_size, output_size, lr=1.0, batchsize=100):
    self._input_size = input_size
    self._output_size = output_size
    self.learning_rate = lr
    self.batchsize = batchsize

    self.W = tf.zeros([input_size, output_size], np.float32)
    self.hb = tf.zeros([output_size], np.float32)
    self.vb = tf.zeros([input_size], np.float32)

  def prob_h_given_v(self, visible, w, hb): 
    return tf.nn.sigmoid(tf.matmul(visible, w) + hb)

  def prob_v_given_h(self, hidden, w, vb):
    return tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(w)) + vb)

  def sample_prob(self, probs):
    return tf.nn.relu(tf.sign(probs - tf.random.uniform(tf.shape(probs))))

  def rbm_reconstruct(self, X):
    h = tf.nn.sigmoid(tf.matmul(X, self.w) + self.hb)
    reconstruct = tf.nn.sigmoid(tf.matmul(h, tf.transpose(self.w)) + self.vb)
    return reconstruct

  def train(self, X, epochs=10):

    loss = []
    for epoch in range(epochs):
      for start, end in zip(range(0, len(X), self.batchsize), range(self.batchsize, len(X), self.batchsize)):
        batch = X[start:end]

        h0 = self.sample_prob(self.prob_h_given_v(batch, self.W, self.hb))
        v1 = self.sample_prob(self.prob_v_given_h(h0, self.W, self.vb))
        h1 = self.sample_prob(self.prob_h_given_v(v1, self.W, self.hb))

        positive_grad = tf.matmul(tf.transpose(batch), h0)
        negative_grad = tf.matmul(tf.transpose(v1), h1)

        self.w = self.W - self.learning_rate * (positive_grad + negative_grad) / tf.dtypes.cast(tf.shape(batch)[0], tf.float32)
        self.vb = self.vb - self.learning_rate * tf.reduce_mean(batch - v1, 0)
        self.hb = self.hb - self.learning_rate * tf.reduce_mean(h0 - h1, 0)

    err = tf.reduce_mean(tf.square(batch - v1))
    print('Epoch: %d' % epoch, 'reconstruction error: %f' % err)
    loss.append(err)

    return loss

(train_data, _), (test_data, _) = tf.keras.datasets.mnist.load_data()
train_data = train_data/np.float32(255)
train_data = np.reshape(train_data, [train_data.shape[0], 784])
test_data = test_data/np.float32(255)
test_data = np.reshape(test_data, [test_data.shape[0], 784])

input_size = train_data.shape[1]
rbm = RBM(input_size, 200)
err = rbm.train(train_data, 50)


out = rbm.rbm_reconstruct(test_data)

row,col = 2,8
idx = np.random.randint(0, 100, row * col // 2)
f, axarr = plt.subplots(row, col, sharex = True, sharey = True, figsize=(20, 4))

for fig, row in zip([test_data, out], axarr):
  for i, ax in zip(idx, row):
    ax.imshow(tf.reshape(fig[i], [28, 28]), cmap='Greys_r')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
