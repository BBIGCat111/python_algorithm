# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 11:30:46 2018

@author: 12100
"""

import numpy as np

#定义一个字典类型的数据结构，建立树结点
# Tab 键值分隔的数据 提取成 列表数据集 成浮点型数据 
class treeNode():
    def _init_(self, feat, val, right, left):
        featureToSplotOn = feat         #特征
        valueOfSplit = val               #值
        rightBranch  = right         #右子树          
        leftBranch = left           #左子树
        
# CART算法实现代码
def loadDataSet(fileName):                 #初始数据处理函数
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))        #将每行映射成浮点数
        dataMat.append(fltLine)
    return dataMat
 
            
# 按特征值 的数据集二元切分  特征(列)  对应的值 
# 某一列的值大于value值的一行样本全部放在一个矩阵里，其余放在另一个矩阵里 
def binSplitDataSet(dataSet, feature, value):        #进行左右区间的区分
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0],:]   #nonzero函数是得到不为0数组中元素的索引，此处为小于value的feature值
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0],:]   # 数组过滤
    return mat0, mat1


# 创建回归树 numpy数组数据集 叶子函数  误差函数  用户设置参数（最小样本数量 以及最小误差下降间隔） 

#回归树的切分函数
# 常量叶子节点 
def regLeaf(dataSet):  # 最后一列为标签 为数的叶子节点 
    return np.mean(dataSet[:, -1])     #mean为求元素的平均值函数   目标变量的均值 
    
# 方差 
def regErr(dataSet):        # 目标变量的平方误差 * 样本个数（行数）的得到总方差 
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]    #var求元素的方差，shape(dataSet)[0]是获得矩阵的行数
                                                                        #shape(dataSet)[1]是获得矩阵的列数

# 选择最优的 分裂属性和对应的大小                                                                              
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]    # 允许的误差下降值 
    tolN = ops[1]    # 切分的最少样本数量
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:     ## 特征剩余数量为1 则返回 
        return None, leafType(dataSet)
    m,n = np.shape(dataSet)    #m为行数，n为列数
    S = errType(dataSet)    # 当前数据集误差 均方误差 
    bestS = np.inf     #刚开始的均方误差设为无穷大
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):    # 遍历 可分裂特征 (列)
        for splitVal in set(dataSet[:, featIndex].T.A.tolist()[0]):   # 遍历对应 特性的 属性值 （行)即同一列不同行
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)  # 进行二元分割
            if(np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):     #样本数量 小于设定值，则不切分 
                continue
            newS = errType(mat0) + errType(mat1)     # 二元分割后的 均方差 
            if newS < bestS:         # 若比分裂前小 则保留这个分类 
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:          # 若分裂后 比 分裂前样本方差 减小的不多 也不进行切分              #如果误差减少一半则退出
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)      #进行切分
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):  #样本数量 小于设定值，则不切分     #如果切分出的数据集很小则退出
        return None, leafType(dataSet)
    return bestIndex, bestValue             # 返回最佳的 分裂属性 和 对应的值 
                
            



def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    # 找到最佳的待切分特征和对应 的值 
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    # 停止条件 该节点不能再分，该节点为叶子节点 
    if feat == None:                 #满足停止条件时返回叶结点值
        return val
    retTree = {}
    retTree['spInd'] = feat   #特征 
    retTree['spVal'] = val   #值
    # 执行二元切分  
    lSet, rSet = binSplitDataSet(dataSet, feat, val)   # 二元切分 左树 右树 
    # 创建左树 
    retTree['left'] = createTree(lSet, leafType, errType, ops)  # 左树 最终返回子叶子节点 的属性值 
    # 创建右树 
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree




#回归树剪枝函数
# 判断是不是树 (按字典形式存储) 
def isTree(obj):
    return (type(obj).__name__=='dict')


# 返回树的平均值 塌陷处理 
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])                                                                        
    return (tree['left'] + tree['right'])/2.0     # 两个叶子节点的 平均值 


# 后剪枝  待剪枝的树  剪枝所需的测试数据 
def prune(tree, testData):
    if np.shape(testData)[0] == 0:
        return getMean(tree)       #没有测试数据 返回 
    if(isTree(tree['right']) or isTree(['left'])):     # 如果回归树的左右两边是树 
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])   #对测试数据 进行切分 
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)        # 对左树进行剪枝 
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)      # 对右树进行剪枝 
    if not isTree(tree['left']) and not isTree(['right']):     #两边都是叶子 
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])     #对测试数据 进行切分 
        errorNoMerge = sum(np.power(lSet[:,-1] - tree['left'], 2)) + sum(np.power(rSet[:,-1] - tree['right'],2))    # 对两边叶子合并前计算 误差   power求平方 
        treeMean = (tree['left'] + tree['right'])/2.0              # 合并后的 叶子 均值          
        errorMerge = sum(np.power(testData[:, -1] - treeMean, 2))     # 合并后 的误差 
        if errorMerge < errorNoMerge:        # 假如合并后的误差小于合并前的误差 
            print("merging")                  # 说明合并后的树 误差更小 
            return treeMean                  
        else:
            return tree
    else:
        return tree




#测量数的结点
def getWidth(tree):
    width = 0
    if isTree(tree):
        width += getWidth(tree['left'])
        width += getWidth(tree['right'])
    else:
        return 1
    return width

#模型树
#模型树的叶结点生成函数
        ######叶子节点为线性模型的模型树######### 
# 线性模型  求系数的函数
def linearSolve(dataSet):
    m, n = np.shape(dataSet)     # 数据集大小 
    X = np.mat(np.ones((m,n)))       # 自变量
    Y = np.mat(np.ones((m,1)))       # 目标变量 
    X[:, 1:n] = dataSet[:, 0:n-1]   # 样本数据集合 
    Y = dataSet[:, -1]      # 标签
    xTx = X.T*X
    if np.linalg.det(xTx) == 0.0:      #行列式值为零,不能计算逆矩阵，可适当增加ops的第二个值    det为计算行列式
        raise NameError('This matrix is singular, cannot do inverse,\n try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)    #.I表示求逆矩阵
    return ws, X, Y              #ws为回归系数


# 模型叶子节点   当全部分类完即到叶结点时调用求回归系数的函数
def modelLeaf(dataSet):
    ws, X,Y = linearSolve(dataSet)
    return ws

# 计算模型误差 
def  modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(np.power(Y - yHat, 2))
        

#用树回归进行预测的代码
    # 模型效果计较 
# 线性叶子节点 预测计算函数 直接返回 树叶子节点 值 
def regTreeEval(model, inDat):
    return float(model)


def modelTreeEval(model, inDat):
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1,n+1)))  # 增加一列 
    X[:, 1:n+1] = inDat
    return float(X * model)    # 返回 值乘以 线性回归系数 



# 树预测函数
def treeForeCast(tree, inData, modelEval = regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)       # 返回 叶子节点 预测值 
    if inData[tree['spInd']] > tree['spVal']:        # 左树 
        if isTree(tree['left']):
            return treeForeCast(tree['left'],inData,modelEval)      # 还是树 则递归调用 
        else:
            return modelEval(tree['left'],inData)    # 计算叶子节点的值 并返回 
    else:
        if isTree(tree['right']):      # 右树 
            return treeForeCast(tree['right'],inData,modelEval)
        else:
            return modelEval(tree['right'],inData)
        
 # 得到预测值            
def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = np.mat(np.zeros((m,1)))    #预测标签 
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat













