import numpy as np
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

K = 5
mnist_data = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training set.
X_training, Y_training = mnist_data.train.next_batch(5500)

# Test set.
X_testing, Y_testing = mnist_data.test.next_batch(1000)

# The training set features.
x_training = tf.placeholder("float", [None, 784])

# The training set labels.
y_training = tf.placeholder("float", [None, 10])

# The testing set features.
x_testing = tf.placeholder("float", [784])

# Euclidean Distance.
distance = tf.negative(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x_training, x_testing)), reduction_indices=1)))

# Prediction: Get min distance neighbors.
values, indices = tf.nn.top_k(distance, k=K, sorted=False)

nearest_neighbors = []
for i in range(K):
    nearest_neighbors.append(tf.argmax(y_training[indices[i]], 0))

neighbors_tensor = tf.stack(nearest_neighbors)
y, idx, count = tf.unique_with_counts(neighbors_tensor)
pred = tf.slice(y, begin=[tf.argmax(count, 0)], size=tf.constant([1], dtype=tf.int64))[0]

accuracy = 0.

# Initializing the variables.
init = tf.initialize_all_variables()

# Launch the graph.
with tf.Session() as sess:
    sess.run(init)

    # loop over test data
    for i in range(len(X_testing)):
        # Get nearest neighbor.
        nn_index = sess.run(pred, feed_dict={x_training: X_training, y_training: Y_training, x_testing: X_testing[i, :]})
        # Get nearest neighbor class label and compare it to its true label.
        print("Test", i, "Prediction:", nn_index,
              "True label:", np.argmax(Y_testing[i]))

        # Calculate accuracy.
        if nn_index == np.argmax(Y_testing[i]):
            accuracy += 1. / len(X_testing)

    print("Done!")
    print("Accuracy:", accuracy)
