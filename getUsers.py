import csv
import numpy as np

filename = 'keystroke.csv'
data = np.loadtxt(open(filename,'rb'),delimiter=',', skiprows=1)	
m,n = data.shape
users = [[] for y in range(56)]
#print len(users)
#print len(users[0])

for i in range(m):
	userNum = int(data[i][0])-1
	##print userNum
	users[userNum].append(data[i][1:n])

print len(users)
print len(users[0])
print len(users[5])
print len(users[55])