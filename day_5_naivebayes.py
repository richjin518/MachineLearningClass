from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def nb_news():

    # 1) get data
    data = fetch_20newsgroups(data_home="D:/DL/", subset="all")

    # 2) data partition
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target)

    # 3) feature engineer: text extraction TfIdf
    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4) naive bayes estimator 
    estimator = MultinomialNB()
    estimator.fit(x_train, y_train)

    # 5) model evaluation
    y_predict = estimator.predict(x_test)
    print("Result compare: ", y_predict == y_test)
    score = estimator.score(x_test, y_test)
    print ("accuracy is: ", score)

    return None


if __name__ == "__main__":
    nb_news()