import numpy as np
import matplotlib.pyplot as plt
import math
import random

#file = open("data.txt")
#data = map( int, file.read().split() )

x = np.arange(0, 100, 0.2)
xArr = []
yArr = []
for i in x:
	lineX = [1]
	lineX.append(i)
	xArr.append(lineX)
	yArr.append( 0.5 * i + 3 + random.uniform(0, 1) * 4 *math.sin(i) )

xMat = np.mat(xArr)
yMat = np.mat(yArr).T
xTx = xMat.T * xMat
if np.linalg.det(xTx) == 0.0:
	print "Can't inverse"
ws = xTx.I * xMat.T * yMat
print ws

y = xMat * ws
#画图
plt.title("linear regression")
plt.xlabel("independent variable")
plt.ylabel("dependent variable")
plt.plot(x, yArr, 'go')
plt.plot(x, y, 'r', linewidth = 2)
plt.show()