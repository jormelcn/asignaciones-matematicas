import pandas as pd

import plotly
from plotly.graph_objs import Histogram
from plotly.graph_objs.histogram import Marker
from plotly.graph_objs.layout import XAxis
from plotly.graph_objs.layout import YAxis
from plotly.graph_objs import Data
from plotly.graph_objs import Figure
from plotly.graph_objs import Layout
from plotly.graph_objs import Bar
from plotly.graph_objs import Scatter
from plotly.graph_objs import Scene
from plotly.graph_objs import Line
import plotly.plotly as py

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

# plotting histograms

traces = []

legend = {0:False, 1:False, 2:False, 3:True}

colors = {
  'Iris-setosa': 'rgb(31, 119, 180)', 
  'Iris-versicolor': 'rgb(255, 127, 14)', 
  'Iris-virginica': 'rgb(44, 160, 44)'
}

for col in range(4):
  for key in colors:
    traces.append(
      Histogram(
        x=X[y==key, col], 
        opacity=0.75,
        xaxis='x%s' %(col+1),
        marker=Marker(color=colors[key]),
        name=key,
        showlegend=legend[col]
      )
    )

layout = Layout(
  barmode='overlay',
  xaxis=XAxis(domain=[0, 0.25], title='sepal length (cm)'),
  xaxis2=XAxis(domain=[0.3, 0.5], title='sepal width (cm)'),
  xaxis3=XAxis(domain=[0.55, 0.75], title='petal length (cm)'),
  xaxis4=XAxis(domain=[0.8, 1], title='petal width (cm)'),
  yaxis=YAxis(title='count'),
  title='Distribution of the different Iris flower features'
)

fig = Figure(data=traces, layout=layout)
py.plot(fig)

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

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('\nEigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

trace1 = Bar(
        x=['PC %s' %i for i in range(1,5)],
        y=var_exp,
        showlegend=False)

trace2 = Scatter(
        x=['PC %s' %i for i in range(1,5)], 
        y=cum_var_exp,
        name='cumulative explained variance')

data = [trace1, trace2]

layout=Layout(
        yaxis=YAxis(title='Explained variance in percent'),
        title='Explained variance by different principal components')

fig = Figure(data=data, layout=layout)
py.plot(fig)


matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1), 
                      eig_pairs[1][1].reshape(4,1)))

print('Matrix W:\n', matrix_w)


Y = X_std.dot(matrix_w)
traces = []

for name in ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'):
    trace = Scatter(
        x=Y[y==name,0],
        y=Y[y==name,1],
        mode='markers',
        name=name,
        marker=Marker(
            size=12,
            line=Line(
                color='rgba(217, 217, 217, 0.14)',
                width=0.5),
            opacity=0.8))
    traces.append(trace)


data = Data(traces)
layout = Layout(showlegend=True,
                scene=Scene(xaxis=XAxis(title='PC1'),
                yaxis=YAxis(title='PC2'),))

fig = Figure(data=data, layout=layout)
py.plot(fig)