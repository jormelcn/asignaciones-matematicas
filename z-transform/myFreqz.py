import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction

#Generar una funcion que devuelva la respuesta en frecuencia del sistema:
a = np.array([1, Fraction(-3,4)], dtype='float_')
b = np.array([1, 0], dtype='float_')

def myFreqz(b, a, w):
  w = np.reshape(w, (len(w), 1))
  ej = np.exp(1.j)
  num = (ej**(w*np.arange(0, len(b)))).dot(b)
  den = (ej**(w*np.arange(0, len(a)))).dot(a)
  return num/den

w = np.linspace(-np.pi, np.pi, 200)
H = myFreqz(b, a, w)
plt.figure()
plt.xlabel('Frecuencia (Radianes)')
plt.ylabel('Ganancia')
plt.title('Respuesta en frecuencia con MyFreqz')
plt.plot(w, np.absolute(H))

# Dado el filtro: H(z) = (z**2 - 1)/(z**2 - sqrt(2)*abs(g)*z + abs(g)**2)
# Grafique la respuesta en frecuencia para: abs(g) = 0.83, 0.96, 0.99

gs = [0.83, 0.96, 0.99]

for g in gs :
  b = np.array([1, -1])
  a = np.array([1, (2**(1/2))*g,  g**2])
  h = myFreqz(b,a, w)
  plt.figure()
  plt.plot(w, np.absolute(h))
  plt.xlabel('Frecuencia (Radianes)')
  plt.ylabel('Ganancia')
  plt.title('Respuesta en frecuencia para g = ' + str(g))

plt.show()



