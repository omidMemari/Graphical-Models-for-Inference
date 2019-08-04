import seaborn as sns
import pandas as pd
import svm_mc as sm
import svm_hmm as sh
import matplotlib.pyplot as plt

sns.set()

def main():
    c_list = [1,10,100,1000]

    #mc_accuracy_wordwise,mc_accuracy_letterwise = sm.run_and_get_accuracy(c_list)
    #sh_accuracy_wordwise,sh_accuracy_letterwise = sh.run_and_get_accuracy(c_list)
    crf_accuracy_wordwise,crf_accuracy_letterwise = ([15.339567905947014, 17.18787693717077, 28.04717917398274, 34.69997709748836],[67.0354754389037547, 68.676950189008438, 75.946205192236115, 80.049058957836584])
    mc_accuracy_wordwise,mc_accuracy_letterwise = ([2.0354754289037547, 7.676650189008438, 14.946205292236115, 16.749054957836584], [48.339567905947014, 61.18787693717077, 68.04717917398274, 69.69997709748836])
    sh_accuracy_wordwise,sh_accuracy_letterwise = ([16.629999999999995, 26.230000000000004, 41.35, 48.27], [67.56622642949843, 75.33781204672113, 82.23910222154362, 84.62478051759676])
    df_list = []
    title_list = []
    mc_df = construct_dataframe_from_result(c_list,mc_accuracy_wordwise,mc_accuracy_letterwise)
    sh_df = construct_dataframe_from_result(c_list,sh_accuracy_wordwise,sh_accuracy_letterwise)
    crf_df = construct_dataframe_from_result(c_list,crf_accuracy_wordwise,crf_accuracy_letterwise)
    df_list.append(mc_df)
    df_list.append(sh_df)
    df_list.append(crf_df)
    title_list.append("Prediction Accuracy vs C for SVM-MC")
    title_list.append("Prediction Accuracy vs C for SVM-HMM")
    title_list.append("Prediction Accuracy vs C for CRF")
    plot_graph(df_list,title_list)

def plot_graph(df_list,title_list):
    f, axes = plt.subplots(1, len(df_list))
    for i in range(0,len(df_list)):
        sns.lineplot(x="c", y="accuracy", hue="type", lw=1,data=df_list[i], ax=axes[i]).set_title(title_list[i])
    #plt.tight_layout()
    plt.show()


def construct_dataframe_from_result(c_list,accuracy_wordwise,accuracy_letterwise):
    dict_list = []
    for i in range(0,len(c_list)):
        m_dict = {}
        m_dict["c"] = c_list[i]
        m_dict["accuracy"] = accuracy_wordwise[i]
        m_dict["type"] = "word_wise_accuracy"
        dict_list.append(m_dict)
        m_dict = {}
        m_dict["c"] = c_list[i]
        m_dict["accuracy"] = accuracy_letterwise[i]
        m_dict["type"] = "letter_wise_accuracy"
        dict_list.append(m_dict)

    return pd.DataFrame(dict_list)







if __name__=="__main__":main()
