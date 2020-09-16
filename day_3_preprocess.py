from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.preprocessing import StandardScaler
"""归一化"""

def scaler_demo():
    data = pd.read_csv('D:/DL/scaler_data.txt') #[[100, 15, 5], [200, 13, 7], [150, 10, 3]]
    scaler = MinMaxScaler(feature_range=(2, 3))
    print(data)
    data_final = scaler.fit_transform(data)
    print(data_final)

def stand_demo():
    data =  pd.read_csv('D:/DL/scaler_data.txt')
    transfer = StandardScaler()
    data_new = transfer.fit_transform(data)
    print(data_new)
    return None

if __name__ == "__main__":
    #scaler_demo()
    stand_demo()
