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
	#print len(users)
	#print len(users[0])
	#print (users[0][0])
	return users

def main():
	filename = 'keystroke.csv'
	users = loadData(filename)

if __name__ == '__main__':
    main()
