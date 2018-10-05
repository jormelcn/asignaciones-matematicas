import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

df = pd.read_csv(
  filepath_or_buffer = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
  header = None,
  sep = ','
)

df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid','class']
df.dropna(how = 'all', inplace = True) # drops the empty line at file-end
df.tail()

# split data table into data X and class labels y

X = df.ix[:,0:4].values
y = df.ix[:,4].values

#standardicing Data
X_std = StandardScaler().fit_transform(X)

#Covariance
cov_mat = np.cov(X_std.T)
print('\nNumPy covariance matrix: \n%s' % cov_mat)

#eigen decomposition
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('\nEigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

#SVD
u,s,v = np.linalg.svd(X_std.T)
print('\nSVD U:')
print(u)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('\nEigenpairs in descending order:')
for i in eig_pairs:
    print(i)

#Cumulative Varianza
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print('\nCumulative Variance in Percent:')
print(cum_var_exp)


#Reducing the 4-dimensional feature space to a 2-dimensional feature subspace, 
#by choosing the "top 2" eigenvectors with the highest eigenvalues to construct our 
#d×k-dimensional eigenvector matrix W
matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1), 
                      eig_pairs[1][1].reshape(4,1)))

print('\nMatrix W:\n', matrix_w)

#Projection Onto the New Feature Space
Y = X_std.dot(matrix_w)
print('\nOriginal Space Shape:', X.shape)
print('New Subspace Shape:', Y.shape)
print('\n')
