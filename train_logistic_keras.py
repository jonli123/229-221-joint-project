from getUsers import retreiveData
from keras.layers import Input, Dense
from keras.models import Model
import tensorflow as tf
import keras.backend as K
from keras.callbacks import LambdaCallback

training_epochs = 100
threshold = 0.5

def logistic_model():
    print("building model")
    # This returns a tensor
    inputs = Input(shape=(142,))

    # a layer instance is callable on a tensor, and returns a tensor
    predictions = Dense(1, activation='sigmoid')(inputs)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def DeepNN(hidden_layers, units_per_layer=10):
    print("building model")
    # This returns a tensor
    inputs = Input(shape=(142,))

    # a layer instance is callable on a tensor, and returns a tensor
    x = Dense(units_per_layer, activation='relu')(inputs)
    for _ in range(1, hidden_layers):
        x = Dense(units_per_layer, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy', recall, precision, false_negatives])
    return model

def train_and_eval(trainX, trainY, testX, testY, model, model_name="keras_DNN"):
    print("Training model")
    model.fit(trainX, trainY, epochs=training_epochs, callbacks=[LambdaCallback(on_epoch_end=clear_local_variables)])  # starts training

    print("Save model to ", model_name)
    model.save_weights(model_name, overwrite=True)

    print("Evaluating Accuracy on validation set:")
    metric_vals = model.evaluate(testX, testY)
    metrics = list(zip(model.metrics_names, metric_vals))
    print("")
    for metric_name, metric_val in metrics:
        print(metric_name, " is ", metric_val)
    predicted = model.predict(testX)
    print("Specificity is ", specificity(testY, predicted > threshold, metric_vals[-1]))
    return metric_vals[1]


def clear_local_variables(epoch, logs=None):
    K.get_session().run(tf.local_variables_initializer())
    #print([(K.get_session().run(i), i.name) for i in tf.local_variables()])

def recall(y_true, y_pred):
    y_hat = tf.cast(y_pred > threshold, tf.float32)
    score, update = tf.metrics.recall(y_true, y_hat)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([update]):
        score = tf.identity(score)
    return score

def precision(y_true, y_pred):
    y_hat = tf.cast(y_pred > threshold, tf.float32)
    score, update = tf.metrics.precision(y_true, y_hat)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([update]):
        score = tf.identity(score)
    return score

def false_negatives(y_true, y_pred):
    y_hat = tf.cast(y_pred > threshold, tf.float32)
    score, update = tf.metrics.false_negatives(y_true, y_hat)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([update]):
        score = tf.identity(score)
    return score


def specificity(y_true, y_pred, false_negatives):
    return (sum(label == 0 for label in y_pred)[0] - false_negatives) / sum(label == 0 for label in y_true)


def main():
    # #load trainX, trainY
    trainX, trainY, testX, testY = retreiveData("full_RUS_undersample")
    metrics = [train_and_eval(trainX, trainY, testX, testY, DeepNN(3), model_name="RUS") for _ in range(1)]
    print(sum(metrics)/len(metrics))

    # # Attack model
    # attacks = np.genfromtxt('data/attack_1mean_all.csv', delimiter=",", skip_header=False)
    # test_attack(attacks)

if __name__ == '__main__':
    main()