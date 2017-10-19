from __future__ import absolute_import
from __future__ import print_function
from data import Unwrapdata ## why does this not work???
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np
#from IPython import embed
#getting data
data = Unwrapdata( subject="S1", roi=1)
X_train, Y_train = data.train_data_set()
X_test, Y_test = data.test_data_set()
X_val, Y_val = data.validation_data_set()


## Nee to do more preprocessing for the images
num_epochs = 100
batch_size = 10
num_steps = X_train.shape[0] / batch_size

cl
# TODO:  make generlized linear regressor (data is fd as input) -flages, argparse
# note need way moe variables then such as shape etc
# TODO:  select different cost functions

def accuracy(predictions, labels):
    return accuracy_score(predcitions, labels)
    


#Defining computation Graph
graph = tf.Graph()
with graph.as_default():

    # input
    tf_X = tf.placeholder(tf.float32, shape=(
        batch_size, X_train.shape[1]), name="X")
    tf_Y = tf.placeholder(tf.float32, shape=(
        batch_size, Y_train.shape[1]), name="Y")
    tf_val_X = tf.constant(X_val, name="X_val")
    tf_test_X = tf.constant(X_test, name="X_test")
    weights = tf.Variable(tf.truncated_normal(
        [X_train.shape[1], Y_train.shape[1]]), name="weights")
    bias = tf.Variable(tf.zeros(Y_train.shape[1]), name="bias")

    predictions = tf.matmul(tf_X, weights) + bias
    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf_Y, logits=predictions,) ## since its mutliclass use this


    opt = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

    train_prediction = predictions
    valid_prediction = tf.matmul(tf_val_X, weights)
    test_prediction = tf.matmul(tf_test_X, weights)

# Need to do computation from here

# Runn session
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for n in range(num_epochs):
        for i in range(num_steps):

            offset = (i * batch_size) % (X_train.shape[0] - batch_size)
            # write unit test - and fully understand what this shizzle doeees
            batch_data = X_train[offset: (offset + batch_size), :]
            batch_labels = Y_train[offset: (offset + batch_size), :]

            feed_dict = {tf_X: batch_data, tf_Y: batch_labels}

            _, l, predictions = session.run(
                [opt, loss, train_prediction], feed_dict=feed_dict)


        if (n % 10 == 0):
            import ipdb; ipdb.set_trace()
            print("Minibatch loss at epoch  %d: %f" % (n, l))
            print("Minibatch accuracy: %f" %
                  accuracy(predictions, batch_labels))
            print("Validation accuracy: %f" % accuracy(
                valid_prediction.eval(), Y_val))
            
            print("Test  accuracy at epoch %d: %f" %
              (n ,accuracy(test_prediction.eval(), Y_test)))
    # gotta write a function here

