import numpy as np
import time
import random
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

# oversample method choice
SMOTE_CONSTANT = 1
ADASYN_CONSTANT = 2

# undersample method choice
RUS_CONSTANT = 3

def loadSingleData(filename):
    data = np.loadtxt(open(filename,'rb'), delimiter=',', skiprows=1)
    m, n = data.shape
    labels = data[:, 0]
    data_points = data[:, 1:]
    return labels, data_points



#loads data with oversampling
def loadPairData(filename, user_cutoff=5, example_cutoff=5, sample_choice = RUS_CONSTANT):
    data = np.loadtxt(open(filename,'rb'),delimiter=',', skiprows=1)
    m,n = data.shape

    pairData = []
    labels = []

    for a_index, a_data in enumerate(data):
        for b_index, b_data in enumerate(data):
            if a_data[0] < user_cutoff and b_data[0] < user_cutoff:
                if a_index % 51 < example_cutoff and b_index % 51 < example_cutoff:
                    if a_index != b_index:
                        # pairData.append((a_index, b_index))
                        concat = np.concatenate((a_data[1:], b_data[1:]))
                        pairData.append(concat)
                        if a_data[0] == b_data[0]: # If they are the same user
                            labels.append(1),
                        else:
                            labels.append(0)

    x_resampled, y_resampled = np.asarray(pairData), np.asarray(labels)

    # oversampling methods, minority class
    if sample_choice == SMOTE_CONSTANT:
        sm = SMOTE(random_state=42)
        x_resampled, y_resampled = sm.fit_sample(pairData, labels)
    elif sample_choice == ADASYN_CONSTANT:
        ada = ADASYN(random_state=42)
        x_resampled, y_resampled = ada.fit_sample(pairData, labels)
    elif sample_choice == RUS_CONSTANT:
        rus = RandomUnderSampler(random_state=42)
        x_resampled, y_resampled = rus.fit_sample(pairData, labels)

    labeledData = list(zip(x_resampled, y_resampled))
    random.shuffle(labeledData)
    cutoff = int(len(labeledData)*0.7)

    trainX, trainY = zip(*labeledData[:cutoff])
    trainX = np.asarray(trainX)
    trainY = np.asarray(trainY)

    testX, testY= zip(*labeledData[cutoff:])
    testX = np.asarray(testX)
    testY = np.asarray(testY)

    return trainX, trainY, testX, testY


def retreiveData(suffix):
    trainX = np.genfromtxt('data/trainX'+suffix+'.csv',delimiter=",",skip_header=False)
    trainY = np.genfromtxt('data/trainY'+suffix+'.csv',delimiter=",",skip_header=False)
    testX = np.genfromtxt('data/testX'+suffix+'.csv',delimiter=",",skip_header=False)
    testY = np.genfromtxt('data/testY'+suffix+'.csv',delimiter=",",skip_header=False)
    len_train = trainX.shape[0]
    len_test = testX.shape[0]

    return trainX, trainY.reshape((len_train, 1)), testX, testY.reshape((len_test, 1))

def main():
    start = time.time()
    filename = 'keystroke.csv'

    '''
    trainX, trainY, testX, testY = loadPairData(filename, user_cutoff=100, example_cutoff=30, sample_choice=ADASYN_CONSTANT)
    suffix = "30_ADASYN_oversample"
    np.savetxt('data/trainX' + suffix + '.csv', trainX, delimiter=',')
    np.savetxt('data/trainY' + suffix + '.csv', trainY, delimiter=',')
    np.savetxt('data/testX' + suffix + '.csv', testX, delimiter=',')
    np.savetxt('data/testY' + suffix + '.csv', testY, delimiter=',')
    '''

    labels, data_points = loadSingleData(filename)

    end = time.time()
    print("Time to run:" + str(end-start))

if __name__ == '__main__':
    main()
    
