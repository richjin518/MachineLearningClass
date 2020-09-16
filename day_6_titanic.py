import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GridSearchCV

def titanic_demo():

    # 1) load data
    path = "http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt"
    data = pd.read_csv(path)

    # 2) prepare feature and label
    x = data[["pclass", "age", "sex"]]
    y = data["survived"]

    #print (x, y)
    # 3) data preprocessing
    # 3.1) missing data 
    x["age"].fillna(x["age"].mean(), inplace=True)
    #print(x["age"])
    # 3.2) when there are class features: one-hot encoding, when multiple featurs: 
    # features -> dict
    x = x.to_dict(orient="records")
    #print(x)

    # 4) partition
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)

    # 5) feature engineering: dict extraction
    transfer = DictVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 6) estimate
    estimator = DecisionTreeClassifier(criterion="entropy")
    #estimator.fit(x_train, y_train)

    # 6-2) Grid search
    param_dict = {"max_depth" : [5, 6, 7, 8, 9, 10, 11]} # different distance metric
    estimator = GridSearchCV(estimator, param_grid = param_dict, cv = 5)
    estimator.fit(x_train, y_train)

    # 7) model evaluation
    y_predict = estimator.predict(x_test)
    print("Result compare: ", y_predict == y_test)
    score = estimator.score(x_test, y_test)
    print ("accuracy is: ", score)

    print("best param: ", estimator.best_params_)
    print("best score: ", estimator.best_score_)
    print("best estimator: ", estimator.best_estimator_)
    print("cv results: ", estimator.cv_results_)


    # export tree
    export_graphviz(estimator.best_estimator_, out_file="D:/DL/titanic_tree.out", feature_names=transfer.get_feature_names())

    return None


if __name__ == "__main__":
    titanic_demo()