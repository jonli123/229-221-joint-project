import csv
import numpy as np

filename = 'keystroke.csv'

data = np.loadtxt(open(filename,'rb'),delimiter=',', skiprows=1)	
m,n = data.shape
out = []
for i in range(n):
	out.append(sum(data[:,i])/m)
	#print sum(data[:,i])/m


print out