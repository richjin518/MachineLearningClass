from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

def decision_tree():
    # 1) get data
    iris = load_iris()

    # 2) partition data
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)

    # 3) feature standardize
    
    # 4) estimate
    estimator = DecisionTreeClassifier(criterion="entropy")
    estimator.fit(x_train, y_train)

    # 5) model evaluation
    y_predict = estimator.predict(x_test)
    print("Result compare: ", y_predict == y_test)
    score = estimator.score(x_test, y_test)
    print ("accuracy is: ", score)

    # export tree
    export_graphviz(estimator, out_file="D:/DL/iris_tree.out", feature_names=iris.feature_names)

    return None


if __name__ == "__main__":
    #variance_filter()
    decision_tree()