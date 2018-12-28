from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Import the data set.
data = input_data.read_data_sets("MNIST_data/",  one_hot=True)

# Training set features.
x = tf.placeholder("float", shape=[None, 784])

# Training set labels.
y_ = tf.placeholder("float", shape=[None, 10])

# Weight value.
W = tf.Variable(tf.zeros([784, 10]))

# Bias.
b = tf.Variable(tf.zeros([10]))

# Build the model.
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Calculate the cost.
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# Change variables to make cost moves in a decreasing direction.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Start the session.
sess = tf.InteractiveSession()

# Initialize the variables.
sess.run(tf.initialize_all_variables())

# Training 1000 times.
for i in range(1000):
    batch = data.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# Test whether the predicted result is equal to the actual result.
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# Calculate the prediction accuracy.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print("Accuracy:", accuracy.eval(feed_dict={x: data.test.images, y_: data.test.labels}))
print("Done!")








