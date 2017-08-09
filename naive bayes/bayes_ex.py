from numpy import *

# 训练集：留言板的中的留言
def create_data_set(): 
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1表示侮辱性语句, 0表示文明用语
    return postingList,classVec
                 
#创建单词表
def create_vocablist(dataSet): 
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #取并集
    return list(vocabSet)

#将输入语句转化为单词频率向量，表示单词表中的哪些单词出现过
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print "the word: %s is not in my Vocabulary!" % word
    return returnVec

#朴素贝叶斯训练函数，输入为所有文档的单词频率向量集合，类标签向量
def trainNB(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix) #文档数量
    numWords = len(trainMatrix[0]) #单词表长度
    pAbusive = sum(trainCategory)/float(numTrainDocs) #侮辱性词语的频率
    p0Num = ones(numWords); p1Num = ones(numWords) #分子初始化为1
    p0Denom = 2.0; p1Denom = 2.0                   #分母初始化为2
    for i in range(numTrainDocs):
        if trainCategory[i] == 1: #如果是侮辱性语句
            p1Num += trainMatrix[i] #矢量相加，将侮辱性语句中出现的词语频率全部加1
            p1Denom += sum(trainMatrix[i]) #屈辱性词语的总量也增加
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom) #变为对数，防止下溢出；对每个元素做除法
    p0Vect = log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive #返回所有词语非侮辱性词语中的频率，所有词语在侮辱性词语中的频率，侮辱性语句的频率

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)#矢量相乘求出概率，log相加
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0
    
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def testingNB():
    listOPosts,classVec = create_data_set()
    myVocabList = create_vocablist(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB(array(trainMat),array(classVec))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
    
#切分文本
def textParse(bigString):    
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 
   
if __name__ == "__main__":
   testingNB()
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   