import numpy as np
import tensorflow as tf
from sys import float_info
from getUsers import retreiveData
# Referenced from https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/logistic_regression.py

learning_rate = 0.001
training_epochs = 25
batch_size = 1024
display_step = 1
threshold = 0.5

# an epsilon to prevent nan loss
epsilon = float_info.epsilon

def train_and_eval(train_x, train_y, test_x, test_y, model_name ="logistic.ckpt"):

    m, n = train_x.shape

    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, n])  # Sequence1 concat Sequence2
    y = tf.placeholder(tf.float32, [None, 1])  # 0 or 1 label

    # Set model weights
    W = tf.Variable(tf.zeros([n, 1]))
    b = tf.Variable(tf.zeros(1))

    # Construct model
    pred = tf.nn.sigmoid(tf.matmul(x, W) + b) # Logistic Regression

    ## TODO we can play with what loss we use
    # Minimize error using cross entropy
    cost = tf.reduce_sum(-(y*tf.log(pred + epsilon) + (1-y)*tf.log(1-pred - epsilon)))
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Initialize the variables (i.e. assign their default value)
    init_g = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init_g)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(m / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs = train_x[i * batch_size:(i + 1) * batch_size]
                batch_ys = train_y[i * batch_size:(i + 1) * batch_size]

                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                              y: batch_ys})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if (epoch + 1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

        print("Training Finished!")
        # Save the variables to disk.
        save_path = saver.save(sess, "./tmp/" + model_name)
        print("Model saved in file: %s" % save_path)

        # Evaluate Model
        # Variables for Evaluation
        yhat = tf.cast(pred > threshold, tf.float32)
        accuracy, acc_update = tf.metrics.accuracy(y, yhat)
        false_pos, fp_update = tf.metrics.false_positives(y, yhat)
        false_neg, fn_update = tf.metrics.false_negatives(y, yhat)
        precision, prec_update = tf.metrics.precision(y, yhat)
        recall, rec_update = tf.metrics.recall(y, yhat)

        sess.run(tf.local_variables_initializer())
        metrics = [accuracy, acc_update,
                   precision, prec_update,
                   recall, rec_update,
                   false_pos, fp_update,
                   false_neg, fn_update]
        sess.run(metrics, feed_dict={x: test_x, y: test_y})

        false_neg_count = sess.run(false_neg)
        specificity = (len(test_y) - false_neg_count)/(sum(label == 0 for label in test_y))

        print("Accuracy: ", sess.run(accuracy))
        print("Precision: ", sess.run(precision)) # True positive / (true positive + false positive)
        print("Recall: ", sess.run(recall)) # True positive / Positive
        print("Specificity: ", specificity) # True negative/ Negative
        print("False Positive Count: ", sess.run(false_pos))
        print("False Negative Count: ", false_neg_count)


    return save_path



def main():
    #load trainX, trainY
    trainX, trainY, testX, testY = retreiveData("full_RUS_undersample")
    train_and_eval(trainX, trainY, testX, testY, "full_RUS")

if __name__ == '__main__':
    main()