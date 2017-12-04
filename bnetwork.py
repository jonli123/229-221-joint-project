import numpy
import csv
'''
def convert_numeric(array):
    for i in range(len(array)):
        array[i] = float(array[i]) '''

def main():
    ''' 
    Read csv file in, as two-dimensional array
    row 0: feature headers
    col 0: subject tags
    '''
    data = numpy.array(list(csv.reader(open('keystroke.csv'))))
    ncols = data.shape[1]
    nfeatures = ncols-1
    
    corr_matrix = [[0.0 for i in range(nfeatures)] for j in range(nfeatures)]
    
    for c1 in range(1, ncols):
        v1 = numpy.array(data[:,c1][1:]).astype(float)
        for c2 in range(1, ncols):
            v2 = numpy.array(data[:,c2][1:]).astype(float)
            corr_matrix[c1-1][c2-1] = numpy.corrcoef(v1, v2)[0][1]
    
    with open('correlation_matrix.csv','wb') as my_csv:
        writer = csv.writer(my_csv, delimiter=',')
        writer.writerows(corr_matrix)

if __name__ == '__main__':
    main()