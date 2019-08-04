import numpy as np

def f(perm, arr, sum, i, j, X, W, T): # recursive function used in brute force algorithm

    if i==len(X)-1:  # last letter does not need T
        arr[i] = j
        sum += np.dot(W[j], X[i])
        perm.append([sum,arr])
        return
    else:
        arr[i] = j
        for k in range(0, len(T)):
            f(perm, arr[:], sum + np.dot(W[j], X[i]) + T[j][k] , i+1, k , X, W, T)

def brute_force(X,W,T): # O(|Y|^m) = O(26^m)
    sum = 0
    i = 0
    j = 0
    arr = [0]*len(X)
    perm = []
    for j in range(len(T)):
        arr[0] = j
        for k in range(len(T)):
            f(perm, arr[:], sum + np.dot(W[j], X[i]) + T[j][k] , i+1, k , X, W, T)

    sum_arr = [perm[i][0] for i in range(len(perm))]
    print("num of operations in brute force(26^m):", len(sum_arr))
    result = perm[np.argmax(sum_arr)]
    return result[1]
    
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
    print("max objective value:", max_sum_value)

    for i in range(m, 0, -1):   # O(m * 26)
        y[i-1] = np.argmax([np.dot(W[j],X[i-1]) + T[j][int(y[i])] + l[i-1][j] for j in range(0,26)])
    
    return y


def print_word(y):  # write the indices in the output file

    output_file = "../result/decode_output.txt"
    with open(output_file, 'w') as f:
        for i in range(len(y)):
            f.write("%s\n" % int(y[i]))
    
    

def str(y):   # create a string based on the letter indices

    result = ''.join( "%s" % chr(int(y[i])+97) for i in range(len(y)))
    return result


def main():

    input_file = "../data/decode_input.txt"
    with open(input_file, 'r') as f:
        data_all = f.readlines()

    start_X = 0
    start_W = start_X + 128*100
    start_T = start_W + 26*128

    X = np.array(data_all[start_X:start_W]).astype(float).reshape(100,128)
    W = np.array(data_all[start_W:start_T]).astype(float).reshape(26,128)
    T = np.array(data_all[start_T:]).astype(float).reshape(26,26)

    z = brute_force(X[:5],W,T) # change the index to compare with max_sum. Note that to fair comparison both indices should be the same.
                               # i.e. brute_force(X[:5],W,T) and max_sum(X[:5],W,T)
    print("brute_force:", str(z))
 
    y = max_sum(X[:],W,T)
    print("max_sum:", str(y))

    print_word(y)

 


if __name__=="__main__":main()
