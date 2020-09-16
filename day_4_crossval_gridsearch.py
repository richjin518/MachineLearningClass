""" model selection and parameter optimization """
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

def model_select_param_opt_demo():
     # 1) get data
    iris = load_iris()

    # 2) partition 
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state = 22)

    # 3) feature engineer: standardize
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test) # use mean and std got in x_train

    # 4) knn with grid search with cross validation
    estimator = KNeighborsClassifier()

    # 5) grid search with cross validation
    param_dict = {"n_neighbors" : [1, 3, 5, 7, 9, 11], "metric" : ['minkowski'], 'p' : [1, 2, 3]} # different distance metric
    estimator = GridSearchCV(estimator, param_grid = param_dict, cv = 10)
    estimator.fit(x_train, y_train)

    # 6) evaluation
    score = estimator.score(x_test, y_test)
    print(score)
    print("best param: ", estimator.best_params_)
    print("best score: ", estimator.best_score_)
    print("best estimator: ", estimator.best_estimator_)
    print("cv results: ", estimator.cv_results_)

    return None


if __name__ == "__main__":
    model_select_param_opt_demo()