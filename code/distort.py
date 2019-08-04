from main import *
import cv2
from matplotlib import pyplot as plt

COLS = 8
ROWS = 16

def main():
    train_filename = "../data/train.txt"
    test_filename = "../data/test.txt"
    #train_data = read_train(train_filename)
    #test_data = read_train(test_filename)

    #original = train_data[0][0][0]
    #rotated = rotate(original ,90)
    #translated = translate(train_data[0][0][0],3,3)

    #show_images(original,rotated,translated)

    num_transformed_list = [500,1000,1500,2000]
    for num_transformed in num_transformed_list:
        data_dict, aux_dict = read_train_data_as_dict()

        get_transformed_data(data_dict, num_transformed)

        num_records = len(data_dict)
        with open("../data/transformed_first"+str(num_transformed),"w") as output_file:
            cur_record = 1
            while(cur_record<=num_records):
                output_file.write(str(cur_record)+" ")
                output_file.write(" ".join(map(str,aux_dict[cur_record])))
                output_file.write(" ")
                output_file.write(" ".join(map(str,data_dict[cur_record])))
                cur_record+=1
                output_file.write("\n")



def read_train_data_as_dict():
    data_dict = {}
    aux_dict = {}
    with open('../data/train.txt') as input_file:
        for line in input_file:
            vals = line.split(" ")
            data_dict[int(vals[0])] = map(int,vals[5:])
            aux_dict[int(vals[0])] = vals[1:5]
    return data_dict,aux_dict




def get_transformed_data(train_dict,num_examples):
    with open("../data/transform.txt") as transform_file:
        for line in transform_file:
            vals = line.strip().split(" ")
            if int(vals[1]) >= num_examples:
                  continue
            if vals[0]=="r":
                train_dict[int(vals[1])] = rotate(np.array(train_dict[int(vals[1])]), float(vals[2])).flatten()
            elif vals[0]=="t":
                train_dict[int(vals[1])] = translate(np.array(train_dict[int(vals[1])]), float(vals[2]), float(vals[3])).flatten()


def rotate(input,degrees):
    original_img = np.uint8(input.reshape(ROWS,COLS))
    rotation_matrix = cv2.getRotationMatrix2D((COLS/2,ROWS/2),90,1)
    rotated_img = cv2.warpAffine(original_img,rotation_matrix,(COLS,ROWS))
    return rotated_img.reshape(1,128)

def translate(input,X_,Y_):
    original_img = np.uint8(input.reshape(ROWS,COLS))
    translation_matrix = np.float32([[1,0,X_],[0,1,Y_]])
    translated_img = cv2.warpAffine(original_img, translation_matrix,(COLS,ROWS))
    return translated_img.reshape(1,128)

def show_images(original,rotated,translated):
    original_img = np.uint8(original.reshape(ROWS,COLS))*255
    rotated_img = np.uint8(rotated.reshape(ROWS,COLS))*255
    translated_img = np.uint8(translated.reshape(ROWS,COLS))*255
    plt.subplot(131),plt.imshow(original_img),plt.title('Original')
    plt.subplot(132),plt.imshow(rotated_img),plt.title('Rotated')
    plt.subplot(133),plt.imshow(translated_img),plt.title('Translated')
    plt.show()










    #for i in range(0,3):
    #    print(train_data[i])




if __name__=="__main__" : main()
