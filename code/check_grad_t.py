import numpy as np
from main import  read_train, read_model
from ref_optimize import gradient_t, get_crf_obj 
from scipy.optimize import check_grad
import math

W, T = read_model()
X_y = read_train("../data/train_small.txt")

def func(t, *args):
        t = t.reshape(26,26)
	w = args[1]
	w = w.reshape(26, 128)
        data = args[0]
	return get_crf_obj(data, np.transpose(w), t, C=1000)

def func_grad(t, *args):
        t = t.reshape(26,26)
	w = args[1]
	w = w.reshape(26, 128)
        data = args[0]
	return gradient_t(data, np.transpose(w), t, C=1000).reshape(26*26)

x0 = W.reshape(26*128)
T = T.reshape(26*26)

print(check_grad(func, func_grad, T, [X_y[0]], x0))
