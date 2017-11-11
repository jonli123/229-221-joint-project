import numpy as np
import tensorflow as tf

# Referenced from https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/logistic_regression.py

learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1
threshold = 0.5

def train(train_x, train_y, test_x, test_y, model_name ="logistic.ckpt"):

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
    cost = tf.reduce_mean(-y*tf.log(pred))
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(m / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs = train_x[i:i + batch_size]
                batch_ys = train_y[i:i + batch_size]

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
        save_path = saver.save(sess, "/tmp/" + model_name)
        print("Model saved in file: %s" % save_path)

        # Evaluate Model
        correct_prediction = tf.equal(pred > threshold, y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy: ", (sess.run(accuracy, feed_dict={x: test_x, y:test_y})))

    return save_path



def main():
	#load trainX, trainY


if __name__ == '__main__':
    main()