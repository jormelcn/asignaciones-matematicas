import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import rigid_regresion as rr

data = pd.read_csv('./prostate.data', delimiter='\t')
print(data)
dataIn = data[['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45']].values
dataOut = data[['lpsa']].values

train = (dataIn[0:50,:], dataOut[0:50,:])
test = (dataIn[50:,:], dataOut[50:,:])

n = 200
params, costHistory = rr.ridge(train, 0.0001, 0.0006, n=n, test = test)[0:2]

plt.plot(np.arange(0, n,1),costHistory)
plt.xlabel('n')
plt.ylabel('Cost')
plt.title('Cost History')
plt.show()

print(params)
