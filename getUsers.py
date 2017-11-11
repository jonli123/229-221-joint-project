import csv
import numpy as np






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
	for i in data:
		for j in data:

			pairData.append((i,j))

	#print len(users)
	#print len(users[0])
	#print (users[0][0])git
	return np.array(users), np.array(pairData)


def main():
	filename = 'keystroke.csv'
	userMatrix,pairData = loadData(filename)

if __name__ == '__main__':
    main()
