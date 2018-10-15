#*********************** Asignacion Matematicas *****************************
# Modulo con funciones para regresion lineal rigida
#
import numpy as np

def verifyShapes(params, dataIn, dataOut):
    if len(dataIn.shape) != 2:
        raise ValueError('dataIn invalid dimensions')
    if len(dataOut.shape) != 2:
        raise ValueError('dataOut invalid dimensions')
    if dataIn.shape[0] != dataOut.shape[0] :
        raise ValueError('dataOut and dataIn have different numbers of elements')
    if len(params.shape) == 1:
        if params.shape[0] != dataIn.shape[1] :
            raise ValueError('number of params are different to dataIn number of columns')
        if dataOut.shape[1] != 1 :
            raise ValueError('number of dataOut colums are different to number of params colums')
    elif len(params.shape) == 2:
        if params.shape[0] != dataIn.shape[1] :
            raise ValueError('number of params rows are different to dataIn number of columns')
        if params.shape[1] != dataOut.shape[1] :
            raise ValueError('number of dataOut colums are different to number of params colums')
    else :
        raise ValueError('params invalid dimensions')

# return (Dstd, mean(D), std(D))
def standardize(D) :
    mean = np.mean(D, axis=0)
    std = np.std(D, axis=0)
    Dstd = (D - mean)/std
    return (Dstd, mean, std)

# d = (Dstd, mean(D), std(D))
# return D
def revertStandardize(d) : 
    D = d[0] * d[2] + d[1]
    return (D, d[1], d[2])

#Medium Relative Cuadratic Error
#return e
def error(Yp, Y):
    err_abs = Yp - Y
    return np.sum(err_abs * err_abs)/np.sum(Y * Y)

# s = (mean(X), std(X), mean(Y), std(Y))
# return Ystd
def predict(tetha, X, s):
    Xstd = (X - s[0])/s[1]
    return Xstd.dot(tetha) * s[3] + s[2]

# diff = X.dot(tetha) - Y
def cost(tetha, diff, delta2) :
    return np.sum(diff * diff) + delta2*np.sum(tetha * tetha)

# diff = X.dot(tetha) - Y
def costDerivate(tetha, diff, delta2, X) :
    return 2 * (diff.T.dot(X) + delta2 * tetha.T).T

def direct(X, Y, delta2) :
    return np.linalg.inv(X.T.dot(X) + delta2 * np.identity(X.shape[1])).dot(X.T).dot(Y)

#return (tetha, success, n, costHistory)
def gradient(
    X, Y, delta2, alpha, tetha = None, 
    n_max = 10000, 
    min_relative_decrement = 1) :
    if tetha == None : tetha = np.zeros([X.shape[1], Y.shape[1]])
    costs = np.zeros(n_max)
    success = False
    for i in range(n_max) :
        diff = X.dot(tetha) - Y
        costs[i] = cost(tetha, diff, delta2)
        if i > 0 : 
            ri = (costs[i] - costs[i - 1])/(alpha*costs[i - 1])
            if ri > 0: break
            if ri <= 0 and ri > - min_relative_decrement :
                success = True
                break
        tetha = tetha - alpha * costDerivate(tetha, diff, delta2, X)
    return (tetha, success, i, costs[0:i])
        
