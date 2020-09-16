import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def facebook_demo():
    # 1) load data
    data = pd.read_csv("D:/DL/facebook.txt")
    #print(data)
    # 2) shrink data range
    # 2.1 by area
    data = data.query("x < 10.5 & x > 1 & y < 9.5 & y > 0.0")
    #print(data)
    # 2.2 by time
    times = pd.to_datetime(data["time"], unit="s")
    #print(times, type(times), times.values, "\n")

    dates = pd.DatetimeIndex(times)
    #print(dates, "\n", dates.weekday, "\n")
    #print(dates.year, "\n", dates.month, "\n", dates.day)
    #print(dates.hour)
    data["day"] = dates.day
    data["weekday"] = dates.weekday
    data["hour"] = dates.hour

    # 2.3 by per place log in times (filter out small amount log in place)
    place_count = data.groupby("place_id").count()["row_id"]
    data_final = data[data["place_id"].isin(place_count[place_count > 1].index.values)]
    
    print(data_final)

    # filter out features and label
    x = data_final[["x", "y", "accuracy", "day", "weekday", "hour"]]
    y = data_final["place_id"]
    print(x, "\n", y)

    # 2.4 parition
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    print(x_train, "\n", x_test)

    # 3 feature engineer: standardize
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4 KNN estimate
    estimator = KNeighborsClassifier()

    param_dict = {"n_neighbors" : [1, 3, 5, 7, 9, 11]}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)
    estimator.fit(x_train, y_train)

    y_predict = estimator.predict(x_test)
    print (y_predict, "\n")
    score = estimator.score(x_test, y_test)
    print("score: ", score)
    print("best param: ", estimator.best_params_)
    print("best score: ", estimator.best_score_)
    print("best estimator: ", estimator.best_estimator_)
    print("cv results: ", estimator.cv_results_)

    return None


if __name__ == "__main__":
    #variance_filter()
    facebook_demo()