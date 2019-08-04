import numpy as np
from string import ascii_lowercase

def read_model():
#function to read model for 2a
    with open("../result/solution.txt", "r") as f:
                raw_data = f.read()
    raw_data = raw_data.split("\n")

    W = np.array(raw_data[:26*128], dtype=float).reshape(26, 128)
    # print "in read_model"
    T = np.array(raw_data[26*128:-1], dtype=float).reshape(26, 26)
    T = np.swapaxes(T, 0, 1)
    return W, T

def read_test():
#function to read testing data
	mapping = list(enumerate(ascii_lowercase))
	mapping = { i[1]:i[0] for i in mapping }

	#with open("../data/test.txt", "r") as f:
	with open("../data/transformed_first2000", "r") as f:
		raw_data = f.read()
	raw_data = raw_data.split("\n")

	dataX, dataY = [], []
	tempX, tempY = [], []
	for row in raw_data[:-1]:
		row = row.split(" ")
		tempY.append( mapping[row[1]])
		tempX.append( np.array(row[5:], dtype=float) )
		if int(row[2]) < 0:
			dataX.append(np.array(tempX))
			dataY.append(np.array(tempY, dtype=int))
			tempX, tempY = [], []

	return dataX, dataY

def max_sum(X,W,T): 

    word_size = len(X)  # 100
    l = np.zeros((word_size,len(T))) # 100 * 26
    y = np.zeros((word_size)) # 100

    for i in range(1, word_size):  # in max-sum algorithm first we store values for l recursively: O(100 * 26 * 26) = O(|Y|m^2)
        for y_i in range(0,26):
            l[i][y_i] = max([np.dot(W[j], X[i-1]) + T[j][y_i] + l[i-1][j] for j in range(0,26)])

###############  recovery part in max-sum algorithm 
         
    m = word_size-1 # 99
    max_sum = [np.dot(W[y_m],X[m]) + l[m][y_m] for y_m in range(0,26)]  # O(26)
    y[m] = np.argmax(max_sum)
    max_sum_value = max(max_sum)
    #print("max objective value:", max_sum_value)

    for i in range(m, 0, -1):   # O(m * 26)
        y[i-1] = np.argmax([np.dot(W[j],X[i-1]) + T[j][int(y[i])] + l[i-1][j] for j in range(0,26)])
    
    return y


def accuracy(y, Y):  # write the indices in the output file
    letter_count = 0
    word_count = 0
    for i in range(len(y)):
        if y[i] == Y[i]:
            letter_count +=1
    if letter_count == len(y):
        word_count = 1
    return letter_count, word_count

def str(y):   # create a string based on the letter indices

    result = ''.join( "%s" % chr(int(y[i])+97) for i in range(len(y)))
    return result


def main():

    W, T = read_model()

    X, Y = read_test()
    
    total_letter_matched = 0.0
    total_word_matched = 0.0
    total_letter = 0
    for i in range(len(Y)):
        y = max_sum(X[i],W,T)
        letter_count, word_count = accuracy(y, Y[i])
        total_letter += len(Y[i])
        total_letter_matched += letter_count
        total_word_matched += word_count
    print "letter accuracy", total_letter_matched/total_letter
    print "word accuracy", total_word_matched/len(Y)
if __name__=="__main__":main()
