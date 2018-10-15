#*********************** Asignacion Matematicas *****************************
# Aplicacion de las funciones de regresion rigida a una base de datos
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importar funciones escritas en ./rigid_regresion.py
import rigid_regresion as rigid

#leer datos
data = pd.read_csv('./prostate.data', delimiter='\t')

#Separar caracteristicas de entrada y salida
X = data[['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45']].values
Y = data[['lpsa']].values

#Separar datos de entrenamiento y de prueba
Xtrain, Ytrain = X[0:50,:], Y[0:50,:]
Xtest, Ytest = X[50:,:], Y[50:,:]

#Estandarizar datos de prueba
Xtrain_std, Xtrain_mean, Xtrain_var = rigid.standardize(Xtrain)
Ytrain_std, Ytrain_mean, Ytrain_var = rigid.standardize(Ytrain)
standardize_const = (Xtrain_mean, Xtrain_var, Ytrain_mean, Ytrain_var)

#Calcular regresion usando gradiente descendiente, alpha = 0.001, d2=0
tetha_gradient, success, n, costs = rigid.gradient(Xtrain_std, Ytrain_std, 0, 0.001)

#Calcular regresion usando funcion directa
tetha_direct = rigid.direct(Xtrain_std, Ytrain_std, 0)

#Predecir las salidas usando los modelos
pred_gradient = rigid.predict(tetha_gradient, Xtrain, standardize_const)
pred_direct = rigid.predict(tetha_direct, Xtrain, standardize_const)

#Calcular el error para los dos metodos
error_gradient = rigid.error(pred_gradient, Ytrain)
error_direct = rigid.error(pred_direct, Ytrain)
print('\nError Gradient: ', error_gradient)
print('Error Direct: ', error_direct)

#Graficar la dinamica del costo de gradiente descendiente
plt.plot(np.arange(0, n,1),costs)
plt.xlabel('n')
plt.ylabel('Cost')
plt.title('Gradient Descent Cost History')

#Calcular multiples regresiones variando d2
delta2 = np.arange(0, 200, 1)
tetha = np.zeros((len(delta2), X.shape[1], Y.shape[1]))
error_train = np.zeros(len(delta2))
error_test = np.zeros(len(delta2))
for i in range(len(delta2)):
  #Calcular regresion usando funcion directa
  tetha[i,:,:] = rigid.direct(Xtrain_std, Ytrain_std, delta2[i])
  #Calcular errores de entrenamiento y de validacion
  pred_train = rigid.predict(tetha[i,:,:], Xtrain, standardize_const)
  pred_test = rigid.predict(tetha[i,:,:], Xtest, standardize_const)
  error_train[i] = rigid.error(pred_train, Ytrain)
  error_test[i] = rigid.error(pred_test, Ytest)
   
#Graficar la variacion de los parametros de la regresion al variar d2
plt.figure()
for i in range(X.shape[1]):
  for j in range(Y.shape[1]):
    plt.plot(delta2, tetha[:,i,j], label='(%d,%d)'%(i,j))
    plt.hold
plt.xlabel('delta2')
plt.ylabel('Lineal Parameters')
plt.title('Regularization')
plt.legend()

#Graficar la variacion del error al variar d2
plt.figure()
plt.plot(delta2, error_train, label='Training Error')
plt.hold
plt.plot(delta2, error_test, label='Testing Error')
plt.xlabel('delta2')
plt.ylabel('Medium Relative Cuadratic Error')
plt.title('Prediction Error')
plt.legend()

plt.show()
