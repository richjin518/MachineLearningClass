from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_demo():
    """Extract TFIDF feature from text"""
    data = ["the lion is king", "long live the king", "lion eat meats"]
    transfer = TfidfVectorizer(stop_words=['is', 'the'])

    data_tfidf = transfer.fit_transform(data)
    print(data_tfidf.toarray())
    print(transfer.get_feature_names())
    return None


if __name__ == "__main__":
    tfidf_demo()