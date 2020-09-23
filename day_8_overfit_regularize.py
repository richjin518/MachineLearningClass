from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.metrics import mean_squared_error

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
    estimator = Ridge()
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
    boston_house_price_ridge_demo_3()
