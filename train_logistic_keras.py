from getUsers import retreiveData
from keras.layers import Input, Dense
from keras.models import Model
import tensorflow as tf
import keras.backend as K
from keras.callbacks import LambdaCallback
import matplotlib.pyplot as plt
import numpy as np
from KmeansAttach import generate_tiled_k_means_attack

training_epochs = 20
threshold = 0.5

def clear_local_variables(epoch, logs=None):
    K.get_session().run(tf.local_variables_initializer())

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
                  metrics=['accuracy', recall, precision, false_negatives])
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

def train_and_eval(trainX, trainY, valX, valY, model, model_name="keras_DNN"):
    print("Training model")
    history = model.fit(trainX, trainY, epochs=training_epochs, validation_data=(valX, valY), batch_size=256,
              callbacks=[LambdaCallback(on_epoch_end=clear_local_variables)])  # starts training

    print("Save model to ", model_name)
    model.save_weights(model_name, overwrite=True)

    print("Plot metrics to ", model_name)
    plot_training_history(history, save_filename=model_name)
    return model, history


def plot_training_history(history, save_filename="model"):

    for metric in ['acc', 'loss', 'recall', 'precision', 'false_negatives']:
        plt.plot(history.history[metric], marker='o', linestyle='--')
        plt.plot(history.history['val_' + metric], marker='o', linestyle='--')
        plt.title(save_filename+' '+ metric)
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(str(save_filename) + "_"+metric+".png")
        plt.close()


def eval_attack(model, attacks):
    predictions = model.predict(attacks)
    successful_attacks = sum(label > 0.5 for label in predictions)
    print("Number of Successful Attacks: ", successful_attacks)
    print("Number of Attacks: ", len(attacks))
    print("Attack Success Rate: ", successful_attacks / len(attacks))
    return successful_attacks, len(attacks), successful_attacks / len(attacks)

def plot_attacks(attack_results, ylabel, ks, model_names, attack_name):
    plt.xscale('log', basex=2)
    # for index, results in enumerate(attack_results):
    #     plt.plot(ks, results, marker='o', linestyle='--')
    lines = [plt.plot(ks, results, marker='o', linestyle='--', label=name) for results, name in zip(attack_results, model_names)]
    plt.legend(loc=2, borderaxespad=0.)
    plt.title(attack_name)
    plt.ylabel(ylabel)
    plt.xlabel('K')
    plt.savefig(str(attack_name)+".png")
    plt.close()



def main():
    # #load trainX, trainY
    # trainX, trainY, valX, valY, testX, testY = retreiveData("RUS_90_5_5")
    # train_and_eval(trainX, trainY, valX, valY, logistic_model(), model_name="Logistic")
    # histories = [train_and_eval(trainX, trainY, valX, valY, Models(i), model_name="DNN_"+str(i))[1] for i in range(1, 11)]
    attack()

def attack():
    ks = [1, 2, 4, 8, 16, 32, 64]
    models = [0, 1, 3, 5, 10]

    k_mean_success_rate = np.zeros((len(models), len(ks)))
    gaussian_success_rate = np.zeros((len(models), len(ks)))
    k_mean_success_count = np.zeros((len(models), len(ks)))
    gaussian_success_count = np.zeros((len(models), len(ks)))
    for k_index, k in enumerate(ks):
        means_attack = generate_tiled_k_means_attack(k, sample_guassian=False, sample_num=1)
        gaussian_attack = generate_tiled_k_means_attack(k, sample_guassian=True, sample_num=5)

        for model_index, model_num in enumerate(models):
            model = DeepNN(model_num)
            model_name = "DNN_" + str(model_num)

            if model_num == 0:
                model = logistic_model()
                model_name = "Logistic"

            model.load_weights("Models/"+model_name)

            print("-------------------------\n")
            print("Attacks on model", model_name)
            print("\nk-mean attack with k =", k)
            num_success, num_attack, success_percentage = eval_attack(model, means_attack)
            k_mean_success_rate[model_index][k_index] = success_percentage
            k_mean_success_count[model_index][k_index] = num_success

            print("\nCluster-based Multivariate Gaussian attack with k =", k, ", 5 sample per distribution")
            num_success, num_attack, success_percentage = eval_attack(model, gaussian_attack)
            gaussian_success_rate[model_index][k_index] = success_percentage
            gaussian_success_count[model_index][k_index] = num_success

    np.savetxt('k_mean_attack_result.csv', k_mean_success_rate, delimiter=',')
    np.savetxt('gaussian_success_rate.csv', gaussian_success_rate, delimiter=',')
    np.savetxt('k_mean_attack_result_count.csv', k_mean_success_count, delimiter=',')
    np.savetxt('gaussian_success_rate_count.csv', gaussian_success_count, delimiter=',')

    model_names = ["Logistic"] + [str(i)+" layer" for i in models[1:]]
    plot_attacks(k_mean_success_rate, "attack success rate", ks, model_names, "K-mean center attack success rate")
    plot_attacks(gaussian_success_rate, "attack success rate", ks, model_names, "K cluster Gaussian attack success rate - 5 sample per cluster")
    plot_attacks(k_mean_success_count, "attack success count", ks, model_names, "K-mean center attack success count")
    plot_attacks(gaussian_success_count, "attack success count", ks, model_names, "K cluster Gaussian attack success count - 5 sample per cluster")

if __name__ == '__main__':
    main()