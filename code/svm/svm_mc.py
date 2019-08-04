#Script for comparing with two alternative approaches:
#multiclass linear SVM on individual letters (SVM-MC),
#and structured SVM (SVM-Struct).

from liblinearutil import *

def convert_to_libsvm_format(input_file_path,output_file_path):
    with open(input_file_path) as input_file , open(output_file_path,'w') as output_file:
        #to try with small subset of data
        # c=0
        # limit=1000
        word_start_indices = []
        for example in input_file:
            # if c==limit :
            #     break
            fields = example.split(" ")
            label = ord(fields[1])-96 #label ord("a")=97 label will be from 1 to 26 corresponding to a to z (only lowercase)
            output_file.write(str(label)+" ")

            if fields[4]==str(1):
                word_start_indices.append(int(fields[0])-1)

            for i in range(5,len(fields)):
                if i==len(fields)-1:
                    output_file.write(str(i-4)+":"+fields[i])
                else:
                    output_file.write(str(i-4)+":"+fields[i]+" ")
        return word_start_indices
            # c+=1

#y, x = svm_read_problem('../../data/train.txt')



def run_and_get_accuracy(c_list):
    input_file_train = '../../data/train.txt'
    output_file_train ='../../data/train_libsvm_format'
    input_file_test = '../../data/test.txt'
    output_file_test ='../../data/test_libsvm_format'

    #Uncomment if running for the first time
    word_start_indices_train = convert_to_libsvm_format(input_file_train, output_file_train)
    word_start_indices_test = convert_to_libsvm_format(input_file_test, output_file_test)


    y_train, x_train = svm_read_problem(output_file_train)
    y_test, x_test = svm_read_problem(output_file_test)
    c_values = map(lambda x: float(x)/len(x_train),c_list)
    accuracy_list_word_wise = []
    accuracy_list_letter_wise = []
    for c in c_values:
        model = train(y_train, x_train, '-c '+str(c))
        p_label, p_acc, p_val = predict(y_test, x_test, model)
        ACC, MSE, SCC = evaluations(y_test, p_label)
        accuracy_list_letter_wise.append(ACC)

        #word-wise accuracy
        incorrect=0
        for i in range(0,len(word_start_indices_test)-1):
            cur_ind = word_start_indices_test[i]
            end_ind = word_start_indices_test[i+1]-1
            while(cur_ind<=end_ind):
                if y_test[cur_ind]!=p_label[cur_ind]:
                    incorrect+=1
                    break
                cur_ind+=1
        word_wise_accuracy = 1 - (float(incorrect)/len(word_start_indices_test))
        accuracy_list_word_wise.append(word_wise_accuracy*100)

    return accuracy_list_word_wise, accuracy_list_letter_wise

def main():
    print(run_and_get_accuracy([1,10,100,1000]))

if __name__=="__main__":main()
