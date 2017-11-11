import csv
import numpy as np
import time





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
			pairData.append((i[1:n],j[1:n]))
			if j[0] == i[0]:
				labels.append(1)
			else:
				labels.append(0)
	print len(labels)
	print len(pairData)
	return np.array(users), np.array(pairData), np.array(labels)


def main():
	start = time.time()

	filename = 'keystroke.csv'
	userMatrix,pairData,labels = loadData(filename)

	end = time.time()
	print "Time to run:" + str(end-start)

if __name__ == '__main__':
    main()
    
