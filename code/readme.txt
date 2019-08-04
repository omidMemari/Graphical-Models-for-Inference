accuracy.py
Code to compute accuracy for letter-wise and word-wise

check_grad_t.py
code to check the gradient w.r.t T against numerical differentiation for Q2(a)

check_grad_w.py
code to check the gradient w.r.t W against numerical differentiation for Q2(a)

distort.py
Code to distort (rotation or translation) the input image

grad_compute_2_a.py
Compute grad and print for q2-a

graph_transformations.py
Code to plot graph for accuray vs # of examples distorted

**********************************************************
inference.py
Code for q 1-c -brute force and max-sum algorithm

y = brute_force(X,W,T)
print(str(y))

brute_force(X,W,T) returns indices for the word maximizing object value. e.g. "123" means the word is "abc". Using str() function we convert it to desired string word. 

y = max_sum(X,W,T)
print(str(y))
print_word(y)

max_sum(X,W,T) also returns indices for the word maximizing object value using dynamic programming and efficiently. Using str() function we convert it to desired string word. Then using print_word(y) we write the indices in the output file.
************************************************************************

inference_2_b.py
Code for inference to compute values for q2-b

main.py
Code to read train, test and model.txt files

ref_optimize.py
Code for forward, backward, logz, gradient_t, gradient_w and objective function

svm/
Code for running SVM-HMM and SVM-MC 
svm_mc.py converts file in appropriate format and calls the windows binary for liblinear package to create the classifier, makes prediction and calulate accuracy
svm_hmm.py calls windows binary for svm_hmm package to createthe classifierm makes prediction and calculates accuracy
plot_graph.py calls svm_mc.py and svm_hmm.py to get wordwise and letterwise accuracy for different c values and plots the result

windows/
contains binary files for liblinear package used to learn SVM-MC
