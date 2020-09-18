import numpy as np

def test_return(list_arr):
    list_arr = [4,5,6]
    return list_arr

def fibo(n, map):
    if n == 1:
        return 1
    if n == 2:
        return 1
    # assume fibo(n-1) 
    left = 0
    if n-1 in map.keys():
        left = map[n-1]
    else:
        left = fibo(n-1, map)
    right = 0
    if n-2 in map.keys():
        right = map[n-2]
    else:
        right = fibo(n-2, map)
    return left + right

def corr():
    a = [0.61, 0.885]
    b = [3.0, 4.0]
    return np.correlate(a, b)

if __name__ == "__main__":
    #map = {}
    a = corr()
    print(a)