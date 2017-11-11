import csv
import numpy as np
import time
import random

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

	for i in data:
		for j in data:
			pairData.append(i[1:n]+j[1:n])
			if j[0] == i[0]:
				labels.append(1)
			else:
				labels.append(0)

	labeledData = zip(pairData,labels)
	#labeledData = pairData + labels
	random.shuffle(labeledData)
	cutoff = int(len(labeledData)*0.7)

	#print cutoff
	#print labeledData[0]

	trainX, trainY = zip(*labeledData[:cutoff])
	#print trainY
	testX,testY= zip(*labeledData[cutoff:])

	#np.savetxt('trainX.csv',trainX,delimiter = ',')
	#np.savetxt('trainY.csv',trainY,delimiter = ',')
	#np.savetxt('testX.csv',testX,delimiter = ',')
	#np.savetxt('testY.csv',testY,delimiter = ',')

	return np.array(trainX), np.array(trainY), np.array(testX), np.array(testY)

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
	print "Time to run:" + str(end-start)

if __name__ == '__main__':
	main()
    
