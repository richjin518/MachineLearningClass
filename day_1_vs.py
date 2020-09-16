
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer

def datasets_demo():
    iris = datasets.load_iris()
    news_data = datasets.fetch_20newsgroups(data_home='D:/DL/',subset='train')
    #print('iris data \n', iris_data)
    #print("description \n", iris_data.DESCR)
    print("features: ", iris.data, iris.data.shape)

    # data split
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    print ("split visualize: " , x_train, x_train.shape)

def dict_demo():
    data = [{'city' : 'Beijing', 'temp':100}, {'city' : 'Shanghai', 'temp':200}, {'city' : 'Beijing', 'temp':300}]
    # Step 1: create extractor instance
    transfer = DictVectorizer(sparse=False)

    # sklearn.feature_extraction
    # sklearn.feature_extraction.DictVectorizer(sparse=True)
    data_new = transfer.fit_transform(data)

    print("data new: ", data_new)
    # one-hot encoding
    print("features new: ", transfer.get_feature_names())
    # DictVectorizer.fit_transform(X) X is the dictionary to be transformed
    # DictVectorizer.get_feature_names() get class names
    return None

def word_count_demo():
    data = ["life is easy", "long live my king", "life is long"]
    transfer = CountVectorizer()
    data_new = transfer.fit_transform(data).toarray()
    print(transfer.get_feature_names())
    print(data_new)

    transfer = CountVectorizer(stop_words=['is'])
    data_new = transfer.fit_transform(data).toarray()
    print(transfer.get_feature_names())
    print(data_new)

    # Chinese tokenize
    data = ["我爱北京天安门", "天安门上太阳升"]
    #jieba.cut(text)

    return None

if __name__ == "__main__":
    #datasets_demo()
    #dict_demo()
    word_count_demo()