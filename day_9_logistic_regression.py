from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

def cancer_tumor_demo():
    # 1) load data and add columns name
    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
    column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
                    'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size',
                    'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
    data = pd.read_csv(path, names = column_names)
    print(len(data), data.isnull().any())

    # 2) data preprocessing: deal with missing values
    data = data.replace(to_replace="?", value=np.nan)
    print(len(data),data.isnull().any())
    data.dropna(inplace = True)
    print(len(data),data.isnull().any())

    # 3) data partition
    # pick feature columns and label column
    x = data.iloc[:, 1:-1]
    y = data["Class"]
    #x_2 = data[column_names[1:10]]
    #print(x[:10], "\n", x_2[:10])
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    # 4) feature engineer: standardize
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 5) estimator
    estimator = LogisticRegression()
    estimator.fit(x_train, y_train)

    # 6) model evaluation
    print("weights: ", estimator.coef_)
    print("b: ", estimator.intercept_)

    y_predict = estimator.predict(x_test)
    print("Result compare: ", y_predict == y_test)
    score = estimator.score(x_test, y_test)
    print ("accuracy is: ", score)

    # precision, recall, F-1 score
    report = classification_report(y_test, y_predict, labels=[2,4], target_names=["good", "bad"])
    print(report)

    # ROC and AUC when sample not even'y distributed
    # TPR = TP / (TP + FN) -> recall
    # FPR = FP / (FP + TN)
    # to calculate AUC 0 -> negative, 1 -> positive  
    # AUC: [0.5 - 1.0]
    y_true = np.where(y_test > 3, 1, 0)
    print(roc_auc_score (y_true, y_predict))
    return None

if __name__ == "__main__":
    #titanic_demo()
    cancer_tumor_demo()
