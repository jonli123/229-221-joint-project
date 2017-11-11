import numpy as np
import time
import random

#loads data in user by user, returns matrix of user data 56x51x71
def loadData(filename, suffix, user_cutoff = 5, example_cutoff = 5):
    data = np.loadtxt(open(filename,'rb'),delimiter=',', skiprows=1)
    m,n = data.shape
    #print (m,n)
    users = [[] for y in range(56)]
    for i in range(m):
        userNum = int(data[i][0])-1
        users[userNum].append(data[i][1:n])

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

    labeledData = list(zip(pairData, labels))
    random.shuffle(labeledData)
    cutoff = int(len(labeledData)*0.7)

    #print cutoff
    # print labeledData[0]

    trainX, trainY = zip(*labeledData[:cutoff])
    trainX = np.asarray(trainX)
    trainY = np.asarray(trainY)
    testX, testY= zip(*labeledData[cutoff:])
    testX = np.asarray(testX)
    testY = np.asarray(testY)


    np.savetxt('data/trainX'+suffix+'.csv',trainX,delimiter = ',')
    np.savetxt('data/trainY'+suffix+'.csv',trainY,delimiter = ',')
    np.savetxt('data/testX'+suffix+'.csv',testX,delimiter = ',')
    np.savetxt('data/testY'+suffix+'.csv',testY,delimiter = ',')

    return np.array(trainX), np.array(trainY), np.array(testX), np.array(testY)

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
    trainX, trainY, testX, testY = loadData(filename, "10user10exp", 10, 10)
    #print len(trainData)
    #print len(testData)
    end = time.time()
    print("Time to run:" + str(end-start))

if __name__ == '__main__':
    main()
    
