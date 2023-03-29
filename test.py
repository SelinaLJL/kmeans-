# -*- coding=utf-8 -*-
# @Time : 2022/3/22 16:30
# @Author : 12137
# @File : test.py
import numpy as np
import time
import matplotlib.pyplot as plt
import kmeans
start_pos=[[7,10],[5,45],[0,20],[25,35],[46,19]]
end_pos=[[30,40],[48,30],[45,2],[25,49],[10,25]]
other_end_pos=[[27,40],[0,0],[5,5],[10,48],[40,4]]
city=start_pos+end_pos+other_end_pos

city=[
    [114.514846,38.059117],
    [114.516916,38.039653],
    [114.545163,38.042697],
    [114.508649,37.990121],
    [114.937240,38.132970],
    [114.956413,38.11077],
    [114.983782,38.065836],
    [115.622507,38.043245],
    [115.618397,38.0010],
    [115.573078,37.971826],
    [115.730829,38.024687],
    [115.767924,38.097153]
]
# print("city:",city)
# city=[[25,20],[5,45],[0,20],[25,1],[40,19],[30,40],[48,30],[45,2],[25,49],[10,25]]
# city=[[0,0],
#       [5.21,8.85],
#       [4.89,7.96],
#       [6.24,0.99],
#       [6.79,2.62],
#       [3.96,3.35],
#       [3.67,6.80],
#       [9.88,1.37],
#       [0.38,7.21],4
#       [2.32,9.13]
#       ]

#如何要聚m类,k设为m+1
k = 4
city=np.array(city)
centroids, clusterAssment = kmeans.kmeans(city, k)  # 调用KMeans文件中定义的kmeans方法。
#clusterAssment是一个（参与聚类的点数）*(2)，每行的第一个参数是类别编号
print(clusterAssment)
## step 3: show the result
print("step 3: show the result...")
kmeans.showCluster(city, k, centroids, clusterAssment)