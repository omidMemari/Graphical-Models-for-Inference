import numpy as np
from main import  read_train, read_model
from ref_optimize import gradient_w, gradient_t, get_crf_obj 
from scipy.optimize import check_grad
import math

W, T = read_model()
data = read_train("../data/train.txt")

w = W.reshape(26, 128)
t = T.reshape(26, 26)

print get_crf_obj(data, np.transpose(w), t, C=1000)/len(data)

#grad_w = gradient_w(data, np.transpose(w), t, C=1000).reshape((26, 128))
#grad_t = gradient_t(data, np.transpose(w), t, C=1000).reshape((26, 26))

#avg_w = np.divide(grad_w, len(data))
#avg_t = np.divide(grad_t, len(data))

#for i in range(26):
#    for j in range(128):
#        print(avg_w[i][j])

#for i in range(26):
#    for j in range(26):
#        print(avg_t[j, i])
