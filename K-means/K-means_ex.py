# encoding: utf-8
#导入testSet中的数据，利用K均值算法对进行分类
from numpy import *

def loadDataSet(fileName):
    dataSet = []
    f = open(fileName)
    for line in f.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)
        dataSet.append(fltLine)
    return mat(dataSet)

#求两个向量的欧式距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))
    
def randCent(dataSet, k):
    n = shape(dataSet)[1] #n是列数
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j]) #找到第j列最小值
        rangeJ = float(max(dataSet[:, j]) - minJ) #求第j列最大值与最小值的差
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1) #生成k行1列的在(0, 1)之间的随机数矩阵
    return centroids
        
def KMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0] #数据集的行
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m): #遍历数据集中的每一行数据
            minDist = inf;minIndex = -1
            for j in range(k): #寻找最近质心
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist: #更新最小距离和质心下标
                    minDist = distJI; minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2 #记录最小距离质心下标，最小距离的平方
        print centroids
        for cent in range(k): #更新质心位置
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]] #获得距离同一个质心最近的所有点的下标，即同一簇的坐标
            centroids[cent,:] = mean(ptsInClust, axis=0) #求同一簇的坐标平均值，axis=0表示按列求均值
    return centroids, clusterAssment
            
import matplotlib.pyplot as plt
def showCluster(dataSet, k, clusterAssment, centroids):
    fig = plt.figure()
    plt.title("K-means")
    ax = fig.add_subplot(111)
    data = []
    for cent in range(k): #提取出每个簇的数据
        ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]] #获得属于cent簇的数据
        data.append(ptsInClust)
    for cent, c, marker in zip( range(k), ['r', 'g', 'b', 'y'], ['^', 'o', '*', 's'] ): #画出数据点散点图
        ax.scatter(data[cent][:, 0], data[cent][:, 1], s=80, c=c, marker=marker)
    ax.scatter(centroids[:, 0], centroids[:, 1], s=1000, c='black', marker='+', alpha=1) #画出质心点
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    plt.show()

def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0] #求所有数据的平均值
    centList =[centroid0]
    for j in range(m): #计算初始误差平方和
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k): #只要聚类的个数小于等于k
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:] #获得属于cent簇的数据
            centroidMat, splitClustAss = KMeans(ptsInCurrCluster, 2, distMeas) #对一个簇中的数据进行kMeans
            sseSplit = sum(splitClustAss[:,1]) #计算本次划分误差平方和
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1]) #计算不属于cent簇的误差平方和
            print "sseSplit, and notSplit: ",sseSplit,sseNotSplit
            if (sseSplit + sseNotSplit) < lowestSSE: #如果有效降低了误差平方和，则记录
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #将划分簇的编号转为新加簇的编号
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit #更新原始的该簇质心编号
        print 'the bestCentToSplit is: ',bestCentToSplit
        print 'the len of bestClustAss is: ', len(bestClustAss)
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#将一个质心更新为两个质心
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss #更新数据的编号以及误差平方和
    return mat(centList), clusterAssment
    

if __name__ == "__main__":
    dataSet = loadDataSet('testSet.txt')
    centroids, clusterAssment = KMeans(dataSet, 4)
    #centroids, clusterAssment = biKmeans(dataSet, 4)
    showCluster(dataSet, 4, clusterAssment,centroids)
    
            
            
            
            