# -*- coding: utf-8 -*-
import numpy as np

# 输入格式如下：
A = np.array([[.5,.2,.3],[.3,.5,.2],[.2,.3,.5]])
B = np.array([[.5,.5],[.4,.6],[.7,.3]])
Pi = np.array([[.2,.4,.4]])
O = np.array([[1,2,1]])

# N = A.shape[0]为数组A的行数， H = O.shape[1]为数组O的列数
# 在下列各函数中，alpha数组和beta数组均为N*H二维数组，也就是横向坐标是时间，纵向是状态

def ForwardAlgo(A, B, Pi, O):
    N = A.shape[0]  # 数组A的行数
    M = A.shape[1]  # 数组A的列数
    H = O.shape[1]  # 数组O的列数

    sum_alpha_1 = np.zeros((M, N))
    alpha = np.zeros((N, H))
    r = np.zeros((1, N))
    alpha_1 = np.multiply(Pi[0, :], B[:, O[0, 0] - 1])

    alpha[:, 0] = np.array(alpha_1).reshape(1,
                                            N)
    # alpha_1是一维数组，在使用np.multiply的时候需要升级到二维数组。#错误是IndexError: too many indices for array

    for h in range(1,H):
        for i in range(N):
            for j in range(M):
                sum_alpha_1[i,j] = alpha[j,h-1] * A[j,i]
            r = sum_alpha_1.sum(1).reshape(1,N)#同理，将数组升级为二维数组
            alpha[i,h] = r[0,i] * B[i,O[0,h]-1]
    print("alpha矩阵: \n %r" % alpha)
    p = alpha.sum(0).reshape(1,H)
    P = p[0,H-1]
    print("观测概率: \n %r" % P)
    #return alpha
    return alpha, P

def BackwardAlgo(A, B, Pi, O):
    N = A.shape[0]  # 数组A的行数
    M = A.shape[1]  # 数组A的列数
    H = O.shape[1]  # 数组O的列数

    # beta = np.zeros((N,H))
    sum_beta = np.zeros((1, N))
    beta = np.zeros((N, H))
    beta[:, H - 1] = 1
    p_beta = np.zeros((1, N))

    for h in range(H - 1, 0, -1):
        for i in range(N):
            for j in range(M):
                sum_beta[0, j] = A[i, j] * B[j, O[0, h] - 1] * beta[j, h]
            beta[i, h - 1] = sum_beta.sum(1)
            # print("beta矩阵: \n %r" % beta)
    for i in range(N):
        p_beta[0, i] = Pi[0, i] * B[i, O[0, 0] - 1] * beta[i, 0]
    p = p_beta.sum(1).reshape(1, 1)
    # print("观测概率: \n %r" % p[0,0])
    return beta, p[0, 0]


def FBAlgoAppli(A, B, Pi, O, I):
    # 计算在观测序列和模型参数确定的情况下，某一个隐含状态对应相应的观测状态的概率
    # 例题参考李航《统计学习方法》P189习题10.2
    # 输入格式：
    # I为二维数组，存放所求概率P(it = qi,O|lambda)中it和qi的角标t和i，即P=[t,i]
    alpha, p1 = ForwardAlgo(A, B, Pi, O)
    beta, p2 = BackwardAlgo(A, B, Pi, O)
    p = alpha[I[0, 1] - 1, I[0, 0] - 1] * beta[I[0, 1] - 1, I[0, 0] - 1] / p1
    return p


def GetGamma(A, B, Pi, O):
    N = A.shape[0]  # 数组A的行数
    H = O.shape[1]  # 数组O的列数
    Gamma = np.zeros((N, H))
    alpha, p1 = ForwardAlgo(A, B, Pi, O)
    beta, p2 = BackwardAlgo(A, B, Pi, O)
    for h in range(H):
        for i in range(N):
            Gamma[i, h] = alpha[i, h] * beta[i, h] / p1
    return Gamma


def GetXi(A, B, Pi, O):
    N = A.shape[0]  # 数组A的行数
    M = A.shape[1]  # 数组A的列数
    H = O.shape[1]  # 数组O的列数
    Xi = np.zeros((H - 1, N, M))
    alpha, p1 = ForwardAlgo(A, B, Pi, O)
    beta, p2 = BackwardAlgo(A, B, Pi, O)
    for h in range(H - 1):
        for i in range(N):
            for j in range(M):
                Xi[h, i, j] = alpha[i, h] * A[i, j] * B[j, O[0, h + 1] - 1] * beta[j, h + 1] / p1
                # print("Xi矩阵: \n %r" % Xi)
    return Xi


def BaumWelchAlgo(A, B, Pi, O):
    N = A.shape[0]  # 数组A的行数
    M = A.shape[1]  # 数组A的列数
    Y = B.shape[1]  # 数组B的列数
    H = O.shape[1]  # 数组O的列数
    c = 0
    Gamma = GetGamma(A, B, Pi, O)
    Xi = GetXi(A, B, Pi, O)
    Xi_1 = Xi.sum(0)
    a = np.zeros((N, M))
    b = np.zeros((M, Y))
    pi = np.zeros((1, N))
    a_1 = np.subtract(Gamma.sum(1), Gamma[:, H - 1]).reshape(1, N)
    for i in range(N):
        for j in range(M):
            a[i, j] = Xi_1[i, j] / a_1[0, i]
            # print(a)
    for y in range(Y):
        for j in range(M):
            for h in range(H):
                if O[0, h] - 1 == y:
                    c = c + Gamma[j, h]
            gamma = Gamma.sum(1).reshape(1, N)
            b[j, y] = c / gamma[0, j]
            c = 0
            # print(b)
    for i in range(N):
        pi[0, i] = Gamma[i, 0]
        # print(pi)
    return a, b, pi


def BaumWelchAlgo_n(A, B, Pi, O, n):  # 计算迭代次数为n的BaumWelch算法
    for i in range(n):
        A, B, Pi = BaumWelchAlgo(A, B, Pi, O)
    return A, B, Pi


def viterbi(A, B, Pi, O):
    N = A.shape[0]  # 数组A的行数
    M = A.shape[1]  # 数组A的列数
    H = O.shape[1]  # 数组O的列数
    Delta = np.zeros((M, H))
    Psi = np.zeros((M, H))
    Delta_1 = np.zeros((N, 1))
    I = np.zeros((1, H))

    for i in range(N):
        Delta[i, 0] = Pi[0, i] * B[i, O[0, 0] - 1]

    for h in range(1, H):
        for j in range(M):
            for i in range(N):
                Delta_1[i, 0] = Delta[i, h - 1] * A[i, j] * B[j, O[0, h] - 1]
            Delta[j, h] = np.amax(Delta_1)
            Psi[j, h] = np.argmax(Delta_1) + 1
    print("Delta矩阵: \n %r" % Delta)
    print("Psi矩阵: \n %r" % Psi)
    P_best = np.amax(Delta[:, H - 1])
    psi = np.argmax(Delta[:, H - 1])
    I[0, H - 1] = psi + 1
    for h in range(H - 1, 0, -1):
        I[0, h - 1] = Psi[I[0, h] - 1, h]
    print("最优路径概率: \n %r" % P_best)
    print("最优路径: \n %r" % I)

ForwardAlgo(A, B, Pi, O)