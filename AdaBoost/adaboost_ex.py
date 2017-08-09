'''
时间：2016/04/19
作者：moverzp
功能：Adaboost的简单例子，使用以下数据学习一个强分类器
数据： 序号   1   2   3   4   5   6   7   8   9   10
        x     0   1   2   3   4   5   6   7   8   9       
        y     1   1   1   -1  -1  -1  1   1   1   -1   
'''
from numpy import *

#创建数据集和标签
def create_dataMat():
    dataMat = mat([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]) #x
    labels = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]
    return dataMat, labels
    
#使用单层决策树对数据进行分类
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq): 
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt': #按照大于或小于关系分类
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0 #所有第dimen维小于等于阈值的分类置为-1
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

#建立最佳单层决策树，即弱分类器    
def buildStump(dataMat,labels,D):
    labelMat = mat(labels).T #转置为列向量
    m,n = shape(dataMat)
    numSteps = 10.0
    bestStump = {} #存贮决策树
    bestClasEst = mat(zeros((m,1))) #最佳单层决策树的结果，初始化为全部分类错误
    minError = inf #最小错误，初始化为无穷大
    for i in range(n): #遍历所有的特征
        rangeMin = dataMat[:,i].min()
        rangeMax = dataMat[:,i].max()
        stepSize = (rangeMax-rangeMin)/numSteps #步长
        for j in range(-1,int(numSteps)+1): #将阈值起始与终止设置在该特征取值的范围之外
            for inequal in ['lt', 'gt']: #取less than和greater than
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMat,i,threshVal,inequal) #使用单层决策树分类
                errArr = mat(ones((m,1))) #1表示分类错误，0表示分类正确，初始化为全部错误
                errArr[predictedVals == labelMat] = 0 #矢量比较，分类正确的置为1
                weightedError = D.T*errArr  #乘以系数D
                #print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError: #如果错误率变小，更新最佳决策树
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst
    
#使用Adaboost建立强分类器，numIt表示最大迭代次数
def adaBoostTrainDS(dataMat,classLabels,numIt=40):
    weakClassArr = []
    m = shape(dataMat)[0]
    D = mat(ones((m,1))/m) #初始化系数D
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataMat,classLabels,D) #建立最佳单层决策树
        #print "D:",D.T
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#计算alpha, max(error,1e-16)防止下溢出
        bestStump['alpha'] = alpha  
        weakClassArr.append(bestStump) #保存弱分类器
        #print "classEst: ",classEst.T
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) #自然底数的指数，为了更新D
        D = multiply(D,exp(expon)) #为下次迭代更新D
        D = D/D.sum()
        aggClassEst += alpha*classEst #矢量相加
        #print "aggClassEst: ",aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1))) #分类正确与错误的结果
        errorRate = aggErrors.sum()/m #分类错误率
        #print "total error: ",errorRate
        if errorRate == 0.0: break #如果分类错误率为0，结束分类
    return weakClassArr
 
#使用Adaboost分类器对数据进行分类 
def adaClassify(datToClass, classifierArr):
    dataMatrix = mat(datToClass) #待分类数据转化为矩阵
    m = shape(dataMatrix)[0] #待分类数据的个数
    aggClassEst = mat(zeros((m,1))) #所有待分类数据的分类，全部初始化为正类
    for i in range(len(classifierArr)): #使用弱分类器，均为矢量运算
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        #print aggClassEst
    return sign(aggClassEst)
    
if __name__ == '__main__':
    dataMat, labels = create_dataMat()
    weakClassArr = adaBoostTrainDS(dataMat, labels)
    for i in range(10):
        res = adaClassify([i], weakClassArr)
        print 'data: %d, class: %2d' % (i, res)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    