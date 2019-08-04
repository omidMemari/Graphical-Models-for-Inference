from string import ascii_lowercase
import numpy as np
from ref_optimize import ref_optimize

def read_train(filename):
#function to read training data
	mapping = list(enumerate(ascii_lowercase))
	mapping = { i[1]:i[0] for i in mapping }

	with open(filename, "r") as f:
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

	ret = zip(dataX, dataY)
	return list(ret)

def read_test(filename):
#function to read testing data
	mapping = list(enumerate(ascii_lowercase))
	mapping = { i[1]:i[0] for i in mapping }

	with open(filename, "r") as f:
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

	ret = zip(dataX, dataY)
	return list(ret)

def read_model():
#function to read model for 2a
    with open("../data/model.txt", "r") as f:
            raw_data = f.read()
    raw_data = raw_data.split("\n")

    W = np.array(raw_data[:26*128], dtype=float).reshape(26, 128)
    # print "in read_model"
    # print W
    T = np.array(raw_data[26*128:-1], dtype=float).reshape(26, 26)
    T = np.swapaxes(T, 0, 1)
    return W, T

def main():
    train_filename = "../data/train_small.txt"
    test_filename = "../data/test.txt"
    train_data = read_train(train_filename)
    test_data = read_train(test_filename)

    # print ref_optimize(train_data, test_data, c=1000)

#main()
