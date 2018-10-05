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

def hypothesis(params, dataIn) :
    return dataIn.dot(params)


def cost(hypothesis, dataOut) :
    difference = hypothesis - dataOut
    return np.sum(difference*difference)/(2 * difference.shape[1])
