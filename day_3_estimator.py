
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def knn_demo():
    # 1) get data
    iris = load_iris()

    # 2) partition 
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state = 6)

    # 3) feature engineer: standardize
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test) # use mean and std got in x_train

    # 4) knn 
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train, y_train)

    # 5) model evaluation
    # method 1: direct compare predicted and real 
    y_predict = estimator.predict(x_test)
    print (y_predict)
    print (y_test == y_predict)

    #method 2: calculate accuracy
    score = estimator.score(x_test, y_test)
    print(score)

    return None

if __name__ == "__main__":
    knn_demo()