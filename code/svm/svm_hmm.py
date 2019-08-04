#Script for comparing with alternative approach
#structured SVM (SVM-Struct).

import os

def run_and_get_accuracy(c_list):
    input_file_train = '../../data/train_struct.txt'
    input_file_test = '../../data/test_struct.txt'

    path_to_executable_learn = 'svm_hmm_learn'
    path_to_executable_classify = 'svm_hmm_classify'
    accuracy_list_word_wise = []
    accuracy_list_letter_wise = []
    for c in c_list:
        os.system(path_to_executable_learn+" -c "+str(c)+" " +input_file_train +" modelfile.dat")
        os.system(path_to_executable_classify+" "+input_file_test+" modelfile.dat predicted_labels.dat > output.txt")

        with open("predicted_labels.dat") as prediction_file, open(input_file_test) as label_file:
            num_correct_letter = 0
            num_letters=0

            for p_line,y_line in zip(prediction_file,label_file):
                y_line_list = y_line.split(" ")
                if p_line.strip()==y_line_list[0]:
                    num_correct_letter+=1
                num_letters+=1
            accuracy_list_letter_wise.append(float(num_correct_letter)/num_letters*100)
        with open('output.txt') as f:
            lines = f.read().splitlines()
            percent_pos = lines[5].find("%")
            accuracy_list_word_wise.append(100-float(lines[5][percent_pos-5 : percent_pos]))

    return accuracy_list_word_wise,accuracy_list_letter_wise

def main():
    print(run_and_get_accuracy([1,10,100,1000]))

if __name__=="__main__":main()
