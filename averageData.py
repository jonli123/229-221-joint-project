from __future__ import print_function
import csv
import numpy as np
import time
import random


def loadData(filename):
	data = np.loadtxt(open(filename,'rb'),delimiter=',', skiprows=1)	
	return data

#generate attacks with average data
def calculateAverages(data):
	m,n = data.shape
	out = []
	pairData = []
	for i in range(1,n):
		out.append(sum(data[:,i])/m)
		#print sum(data[:,i])/m

	for i , row in enumerate(data):
		concat = np.concatenate((out, row[1:]))
		pairData.append(concat)
		concat = np.concatenate((row[1:], out))
		pairData.append(concat)

	print(out)
	print(len(out)) #71
	print(len(pairData))#5712
	print(len(pairData[0])) #142
	return np.array(pairData)

def main():
	start = time.time()
	filename = 'keystroke.csv'
	data = loadData(filename)
	attacks = calculateAverages(data)
	np.savetxt('data/attack_1mean_all.csv', attacks, delimiter=',')
	end = time.time()
	print("Time to run:" + str(end-start))

if __name__ == '__main__':
    main()