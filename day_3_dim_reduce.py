"""
Filter 
    方差小
    相关系数： feature vs feature relevance
    Pearson Correlation Coefficient | 0.4 | 0.7 |
        强相关性
            选其中一个
            加权求和
            主成分分析 （PCA）
"""

from sklearn.feature_selection import VarianceThreshold
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

"""
filter low variance feature
"""
def variance_filter():
    data = pd.read_csv('D:/DL/scaler_data.txt') 
    print(data)
    #data = data.iloc[:, 0:-1]
    #print(data)

    filter = VarianceThreshold(threshold=1)
    data_new = filter.fit_transform(data)
    print(data_new, data_new.shape)

    # calculate two features correlation
    r = pearsonr(data["A"], data["B"])
    print(r)

    plt.figure(figsize=(20, 8), dpi=100)
    plt.scatter(data["A"], data["B"])
    plt.show()

    return None

def pca_demo():
    """parameter: n_components= if integer : num of dims/features, if float : percentage of info to keep"""
    data = [[2,8,4,5], [1,7,3,8], [5,4,9,1]]
    # new instance
    transfer = PCA(n_components=0.95)
    data_new = transfer.fit_transform(data)
    print (data_new)

    # pd.merger(table1, table2, on=["table1 key col", "table2 key col"])
    # pd.crosstab(table["col1", table["col2"]])

    return None

if __name__ == "__main__":
    #variance_filter()
    pca_demo()