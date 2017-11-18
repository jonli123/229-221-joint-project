import numpy as np
import tensorflow as tf
from sys import float_info
from getUsers import retreiveData
from keras.layers import Input, Dense
from keras.models import Model

training_epochs = 25
batch_size = 128

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
                  metrics=['accuracy'])
    return model

def train_and_eval(trainX, trainY, testX, testY, model, model_name="keras_DNN"):
    print("Training model")
    model.fit(trainX, trainY, epochs=training_epochs)  # starts training

    print("Save model to ", model_name)
    model.save_weights(model_name, overwrite=True)

    print("Evaluating Accuracy on validation set:")
    metric_vals = model.evaluate(testX, testY)
    metrics = zip(model.metrics_names, metric_vals)
    print("")
    for metric_name, metric_val in metrics:
        print(metric_name, " is ", metric_val)


def main():
    # #load trainX, trainY
    trainX, trainY, testX, testY = retreiveData("full_RUS_undersample")
    train_and_eval(trainX, trainY, testX, testY, DeepNN(2), model_name="RUS")

    # # Attack model
    # attacks = np.genfromtxt('data/attack_1mean_all.csv', delimiter=",", skip_header=False)
    # test_attack(attacks)

if __name__ == '__main__':
    main()