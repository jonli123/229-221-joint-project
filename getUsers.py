import numpy as np
import time
import random
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler


user_cutoff = 2
example_cutoff = 2

#loads data in user by user, returns matrix of user data 56x51x71
def loadData(filename):
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
                        pairData.append(a_data[1:] + b_data[1:])
                        if a_data[0] == b_data[0]: # If they are the same user
                            labels.append(1),
                        else:
                            labels.append(0)

    labeledData = zip(pairData,labels)
    random.shuffle(labeledData)
    cutoff = int(len(labeledData)*0.7)

    #print cutoff
    # print labeledData[0]

    trainX, trainY = zip(*labeledData[:cutoff])
    testX, testY= zip(*labeledData[cutoff:])
    
    np.savetxt('trainX.csv',trainX,delimiter = ',')
    np.savetxt('trainY.csv',trainY,delimiter = ',')
    np.savetxt('testX.csv',testX,delimiter = ',')
    np.savetxt('testY.csv',testY,delimiter = ',')
    
    # oversample method choice
    SMOTE_CONSTANT = 1
    ADASYN_CONSTANT = 2
    
    # undersample method choice
    RUS_CONSTANT = 1
    
    
    oversample_choice = SMOTE_CONSTANT
    undersample_choice = RUS_CONSTANT
    
    # oversampling methods, minority class    
    if oversample_choice == SMOTE_CONSTANT:
        sm = SMOTE(random_state=42)
        trainX_resampled, trainY_resampled = sm.fit_sample(trainX, trainY)
        return np.array(trainX_resampled), np.array(trainY_resampled), np.array(testX), np.array(testY)
    
    if oversample_choice == ADASYN_CONSTANT:
        ada = ADASYN(random_state=42)
        trainX_resampled, trainY_resampled = ada.fit_sample(trainX, trainY)
        return np.array(trainX_resampled), np.array(trainY_resampled), np.array(testX), np.array(testY)
    
    # undersampling methods, majority class
    if undersample_choice == RUS_CONSTANT:
        rus = RandomUnderSampler(random_state=42)
        trainX_resampled, trainY_resampled = rus.fit_sample(trainX, trainY)
        return np.array(trainX_resampled), np.array(trainY_resampled), np.array(testX), np.array(testY)        
    
    #return np.array(trainX), np.array(trainY), np.array(testX), np.array(testY)
    # return np.array(trainX_resampled), np.array(trainY_resampled), np.array(testX), np.array(testY)

def retreiveData():
    trainX = np.genfromtxt('trainX.csv',delimiter=",",skip_header=False)
    trainY = np.genfromtxt('trainY.csv',delimiter=",",skip_header=False)
    testX = np.genfromtxt('testX.csv',delimiter=",",skip_header=False)
    testY = np.genfromtxt('testY.csv',delimiter=",",skip_header=False)
    return trainX, trainY, testX, testY

def main():
    start = time.time()

    filename = 'keystroke.csv'
    trainX, trainY, testX, testY = loadData(filename)
    #print len(trainData)
    #print len(testData)
    end = time.time()
    print("Time to run:" + str(end-start))

if __name__ == '__main__':
    main()
    
