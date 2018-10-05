import numpy as np
import gradient_descent as gd

params = np.array([
    [0.1, 1], 
    [0.2, 2],
    [0.1, 1],
    [0.2, 2]
])
dataIn = np.array([
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
])
dataOut = np.array([
    [1, 2],
    [1, 2],
    [1, 2],
    [1, 2],
    [1, 2],
    [1, 2]
])
gd.verifyShapes(params, dataIn, dataOut)
cost = gd.cost(gd.hypothesis(params, dataIn), dataOut)
print('Cost: ', cost)
