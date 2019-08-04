import matplotlib.pyplot as plt
import seaborn as sns


x = [0,500,1000,1500,2000]

word_wise_accuracy = [16.86, 16.22, 15.35, 14.01, 13.57]
letter_wise_accuracy = [69.72, 69.15, 68.48, 67.65, 67.10]

word_wise_accuracy_crf = [16.86, 16.22, 15.35, 14.01, 13.57]
letter_wise_accuracy_crf = [69.72, 69.15, 68.48, 67.65, 67.10]



   
ax = sns.lineplot(x,word_wise_accuracy).set_title("SVM-MC word wise accuracy vs no. of distorted training examples")
plt.xlabel("no. of distorted examples")
plt.ylabel("accuracy")
plt.show()

ax2 = sns.lineplot(x,letter_wise_accuracy).set_title("SVM-MC letter wise accuracy vs no. of distorted training examples")
plt.xlabel("no. of distorted examples")
plt.ylabel("accuracy")

plt.show()

ax2 = sns.lineplot(x,letter_wise_accuracy).set_title("CRF letter wise accuracy vs no. of distorted training examples")
plt.xlabel("no. of distorted examples")
plt.ylabel("accuracy")

plt.show()

ax2 = sns.lineplot(x,letter_wise_accuracy).set_title("CRF letter wise accuracy vs no. of distorted training examples")
plt.xlabel("no. of distorted examples")
plt.ylabel("accuracy")

plt.show()