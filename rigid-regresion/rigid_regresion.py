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

def standarice(data) :
    mean = np.mean(data, axis=0)
    variance = np.std(data, axis=0)
    dataStd = (data - mean)/variance
    return (dataStd, mean, variance)

def calcDifference(params, dataIn, dataOut) :
    return dataIn.dot(params) - dataOut

def calcError(
    params, testIn, testOut, 
    dataInMean, dataInVar, 
    dataOutMean, dataOutVar) :
    prediction = ((testIn - dataInMean)/dataInVar).dot(params)*dataOutVar + dataOutMean
    difference = prediction - testOut
    return np.sum(difference*difference, axis=0)

def basicCost(difference) :
    return difference.dot(difference.T)/(2 * difference.shape[1])

def basicCostDerivate(difference, dataIn) :
    return difference.T.dot(dataIn).T/difference.shape[1]

def basicGradientDescent(alpha, params,difference, dataIn) :
    return params - alpha * basicCostDerivate(difference, dataIn)

def cost(delta2, params, difference) :
    return basicCost(difference) + delta2*np.sum(params*params)

def costDerivate(delta2, params, difference, dataIn) :
    return 2*(difference.T.dot(dataIn) + delta2*params.T).T

def gradientDescent(delta2, alpha, params, diference, dataIn) :
    return params - alpha * costDerivate(delta2, params, diference, dataIn)

def ridge(train, test, delta2, alpha, params = None, n = 1000, trainErrorHistory = False, testErrorHistory = False) :
    dataIn, dataOut = train
    testIn, testOut = test
    if params == None : params = np.zeros([dataIn.shape[0], dataOut.shape[0]]) 
    verifyShapes(params, dataIn, dataOut)
    verifyShapes(params, testIn, testOut)
    dataInStd, dataInMean, dataInVar = standarice(dataIn)
    dataOutStd, dataOutMean, dataOutVar = standarice(dataOut)
    diference = calcDifference(params, dataInStd, dataOutStd)
    costHistory = []
    trainErrorHistory = []
    testErrorHistory = []
    for _ in range(n) :
        params = gradientDescent(delta2, alpha, params, diference, dataIn)
        diference = calcDifference(params, dataInStd, dataOutStd)
        currentCost = cost(delta2, params, diference)
        costHistory.append(currentCost)
        if trainErrorHistory :
            trainError = calcError(
                params, dataIn, dataOut,
                dataInMean, dataInVar,
                dataOutMean, dataOutVar
            )
            trainErrorHistory.append(trainError)
        if testErrorHistory :
            testError = calcError(
                params, testIn, testOut, 
                dataInMean, dataInVar, 
                dataOutMean, dataOutVar)
            testErrorHistory.append(testError)
    return (params, costHistory, trainErrorHistory, testErrorHistory)
        
