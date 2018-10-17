#*********************** Asignacion Matematicas *****************************
# Modulo con funciones para regresion lineal lasso
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
    return np.sum(diff * diff) + delta2*np.sum(np.abs(tetha))

def iterar(tetha, X, Y, delta2):
    mask = np.ones(X.shape[1], dtype='bool')
    for j in range(X.shape[1]):
        mask[j] = False
        a = 2*np.sum(X[:,j]**2)*tetha[j,:]
        c = 2*np.sum((Y - X[:,mask].dot(tetha[mask,:]))*X[:,j:j+1], axis=0)
        mask[j] = True
        tetha[j,:] = ((c < -delta2)*(c + delta2) + (c > delta2)*(c - delta2))/a
    return tetha

def regresion(X, Y, delta2, tetha = None, n_max=10000, t = 0.00001):
    if tetha == None : tetha = np.random.random((X.shape[1], Y.shape[1]))
    success = False
    for n in range(n_max):
        tetha_n = iterar(np.copy(tetha), X, Y, delta2)
        if np.sum(np.abs(tetha - tetha_n)) < t:
            success = True
            break
        tetha = tetha_n
    return (tetha, success, n)
