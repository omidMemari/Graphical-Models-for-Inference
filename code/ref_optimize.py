# Imports
import numpy as np
import scipy.optimize as opt
from math import log

def forward(X, m, W, T):
    # T = np.transpose(T)
    alpha = np.zeros((m, 26))
    for i in range(1, m):
        for j in range(26):
            total_sum = []
            for k in range(26):
                 total_sum.append(np.dot(W[k], X[i-1]) + T[k,j] + alpha[i-1, k])
            temp = log_sum_exp(np.array(total_sum))
            alpha[i, j] = temp
    return alpha

def backward(X, m, W, T):
    beta = np.zeros((m, 26))
    # T = np.transpose(T)
    for i in range(m-2, -1, -1):
        for j in range(26):
            total_sum = []
            for k in range(26):
                total_sum.append(np.dot(W[k], X[i+1]) + T[j,k] + beta[i+1, k])
            temp = log_sum_exp(np.array(total_sum))
            beta[i, j] = temp
    return beta    

def log_sum_exp(arr):
    M = arr.max()
    return log(np.sum(np.exp(np.add(arr, -1*M)))) + M

def calculate_log_z(X, m, W, T):
    alpha = forward(X, m, W, T)
    z = []
    for i in range(26):
        z.append(np.add(np.dot(W[i], X[m-1]), alpha[m-1, i]))
    return log_sum_exp(np.array(z))

def gradient_w(word_list, W, T, C):
    
    grad_w = np.zeros((26, 128))
    indicator = np.zeros(26)
    W_t = np.transpose(W)
    for X, Y in word_list:
        m = len(Y)
        alpha = forward(X, m, W_t, T)
        beta = backward(X, m, W_t, T)
        log_z = calculate_log_z(X, m, W_t, T)
        temp_grad = np.zeros((26, 128))
        for s in range(m):
            prob = np.add(alpha[s,:], beta[s,:])
            # node = np.matmul(np.transpose(W), X[s])
            # node = np.matmul(W[Y[s]], X[s])
            node = np.matmul(W_t, X[s])
            prob = np.add(prob, node)
            prob = np.add(prob, -1*log_z)
            prob = np.exp(prob)

            indicator[Y[s]] = 1
            indicator = np.add(indicator, -1*prob)
            letter_grad = np.tile(X[s], (26, 1))
            out = np.multiply(indicator[:, np.newaxis], letter_grad)
            temp_grad = np.add(out, temp_grad)
            indicator[:] = 0
        grad_w = np.add(grad_w, temp_grad)
    grad_w = np.multiply(grad_w, -1*C/len(word_list))
    grad_w = np.add(grad_w, W_t)
    return grad_w

def gradient_t(word_list, W, T, C):
    grad_t = np.zeros((26, 26))
    indicator = np.zeros((26, 26))
    W_t = np.transpose(W)
    for X, Y in word_list:
        m = len(Y)
        alpha = forward(X, m, W_t, T)
        beta = backward(X, m, W_t, T)
        log_z = calculate_log_z(X, m, W_t, T)
        temp_grad = np.zeros((26, 26))
        for s in range(m-1):
            node = np.add.outer(np.matmul(W_t, X[s]), np.matmul(W_t, X[s+1]))
            node = np.add(node, T)
	    node = np.add(alpha[s][:, np.newaxis], node)
            node = np.add(beta[s+1], node)
            prob = np.add(-1*log_z, node)
            prob = np.exp(prob)

            indicator[Y[s], Y[s+1]] = 1
            out = np.add(indicator, -1*prob)
            temp_grad = np.add(out, temp_grad)
            indicator[:, :] = 0
        grad_t = np.add(grad_t, temp_grad)
    grad_t = np.multiply(grad_t, -1*C/len(word_list))
    grad_t = np.add(grad_t, T)
    return grad_t
    # return temp_grad

def get_crf_obj(word_list, W, T, C):
    log_likelihood = 0.0
    W_t = np.transpose(W)
    n = len(word_list)
    #log-likelihood calculation
    for X,Y in word_list:
        z = calculate_log_z(X, len(Y), W_t, T)
        z_x = z

        node_poten = 0.0
        edge_poten = 0.0
        for s in range(len(Y)):
                y_s = Y[s]
                node_poten += np.dot(W_t[y_s], X[s])
        for s in range(len(Y)-1):
            edge_poten += T[Y[s]][Y[s+1]]
        p_y_x = node_poten + edge_poten - z_x
        log_likelihood += p_y_x

    # norm_w calculation
    norm_w = [] 
    for i in range(26):
        norm_w.append(np.linalg.norm(W_t[i]))
    norm_w = np.sum(np.square(norm_w))

    #norm_t calculation
    norm_t = np.sum(np.square(T))
    return -1*(C/n)*log_likelihood + 0.5 * norm_w + (0.5 * norm_t)
    #return log_likelihood
    

def crf_obj(x, word_list, C):
    """Compiute the CRF objective and gradient on the list of words (word_list)
    evaluated at the current model x (w_y and T, stored as a vector)
    """
    
    # x is a vector as required by the solver. So reshape it to w_y and T
    W = np.reshape(x[:128*26], (128, 26))  # each column of W is w_y (128 dim)
    T = np.reshape(x[128*26:], (26, 26))  # T is 26*26

    f = get_crf_obj(word_list, W, T, C)  # Compute the objective value of CRF
                                         # objective log-likelihood + regularizer

    g_W = gradient_w(word_list, W, T, C)                  # compute the gradient in W(128 * 26)
    g_T = gradient_t(word_list, W, T, C)                  # compute the gradient in T(26*26)
    #g_W = 1
    #g_T = 1
    g = np.concatenate([g_W.reshape(-1), g_T.reshape(-1)])  # Flatten the
                                                          # gradient back into
                                                          # a vector
    return [f,g]

def crf_test(x, word_list):
    """
    Compute the test accuracy on the list of words (word_list); x is the
    current model (w_y and T, stored as a vector)
    """

    # x is a vector. so reshape it into w_y and T
    W = np.reshape(x[:128*26], (128, 26))  # each column of W is w_y (128 dim)
    T = np.reshape(x[128*26:], (26, 26))  # T is 26*26

    # Compute the CRF prediction of test data using W and T
    y_predict = crf_decode(W, T, word_list)

    # Compute the test accuracy by comparing the prediction with the ground truth
    accuracy = compare(y_predict, true_label_of_word_list)
    print('Accuracy = {}\n'.format(accuracy))

def ref_optimize(train_data, test_data, C):
    #assert False
    print('Training CRF ... c = {} \n'.format(C))

    # Initial value of the parameters W and T, stored in a vector
    x0 = np.zeros((128*26+26**2,1))

    # Start the optimization
    #x = crf_obj(x0, train_data, c)
    result = opt.fmin_tnc(crf_obj, x0, args = [train_data, C], maxfun=100,
                          ftol=1e-3, disp=5)
    model  = result[0]          # model is the solution returned by the optimizer
    #accuray = 1
    accuracy = crf_test(model, test_data)
    print('CRF test accuracy for c = {}: {}'.format(c, accuracy))
    return accuracy





    
    
