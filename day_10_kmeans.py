import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

order_products = pd.read_csv('D:/DL_data/instacart-market-basket-analysis/order_products__prior/order_products__prior.csv')
products = pd.read_csv("D:/DL_data/instacart-market-basket-analysis/products/products.csv")
orders = pd.read_csv("D:/DL_data/instacart-market-basket-analysis/orders/orders.csv")
aisles = pd.read_csv("D:/DL_data/instacart-market-basket-analysis/aisles/aisles.csv")

table1 = pd.merge(aisles, products, on=["aisle_id", "aisle_id"])
table2 = pd.merge(table1, order_products, on=["product_id", "product_id"])
table3 = pd.merge(table2, orders, on=["order_id", "order_id"])

#print (table3)
table = pd.crosstab(table3["user_id"], table3["aisle"])

data = table[:10000]

transformer = PCA(n_components=0.95)
data_new = transformer.fit_transform(data)

estimator = KMeans(n_clusters=3)
estimator.fit(data_new)

# clustering evaluation metric: silhouette coefficient: worst [-1, 1] best
y_predict = estimator.predict(data_new)
score = silhouette_score(data_new, y_predict)

print(score)



