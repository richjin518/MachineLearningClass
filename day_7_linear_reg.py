from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.metrics import mean_squared_error

def boston_house_price_demo_1():
    """Solve equation system"""

    # 1) load data
    boston = load_boston()
    print ("features: ", boston.data.shape)
    # 2) parition
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)

    # 3) feature engineer:
    # multiple featuers -> dimentionless -> Standardize (robust to abnomal data point)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4) estimator
    # fit
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)

    # 5) print model
    print("weights: ", estimator.coef_)
    print("b: ", estimator.intercept_)

    # 6) model evaluation
    y_predict = estimator.predict(x_test)
    error = mean_squared_error(y_test, y_predict)
    print(error)

    return None

def boston_house_price_demo_2():
    """Gridient descent"""

    # 1) load data
    boston = load_boston()

    # 2) parition
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)

    # 3) feature engineer:
    # multiple featuers -> dimentionless -> Standardize (robust to abnomal data point)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4) estimator
    # fit
    estimator = SGDRegressor(eta0=0.0001, max_iter=10000000, penalty="l1")
    estimator.fit(x_train, y_train)

    # 5) model evaluation
    print("weights: ", estimator.coef_)
    print("b: ", estimator.intercept_)

     # 6) model evaluation
    y_predict = estimator.predict(x_test)
    error = mean_squared_error(y_test, y_predict)
    print(error)

    return None

def boston_house_price_ridge_demo_3():
    """Gridient descent"""

    # 1) load data
    boston = load_boston()

    # 2) parition
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)

    # 3) feature engineer:
    # multiple featuers -> dimentionless -> Standardize (robust to abnomal data point)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4) estimator
    # fit
    estimator = Ridge(max_iter=10000, alpha=0.3)
    estimator.fit(x_train, y_train)

    # 5) model evaluation
    print("weights: ", estimator.coef_)
    print("b: ", estimator.intercept_)

     # 6) model evaluation
    y_predict = estimator.predict(x_test)
    error = mean_squared_error(y_test, y_predict)
    print(error)

    return None

if __name__ == "__main__":
    #titanic_demo()
    boston_house_price_demo_1()
    boston_house_price_demo_2()
    boston_house_price_ridge_demo_3()