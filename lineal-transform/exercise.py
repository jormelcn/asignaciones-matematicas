import numpy as np

def isLineal(L, V, W, e_max=1e-5, r = 1000):
  a = np.random.random((100, V))*r
  b = np.random.random((100, V))*r
  r1 = np.apply_along_axis(L, 1, a) + np.apply_along_axis(L, 1, b)
  r2 = np.apply_along_axis(L, 1, a + b)
  e = np.max(np.abs(r1 - r2))
  if e > e_max : 
    return False
  c = np.random.random(100)*r
  for i in range(len(c)):
    r3 = np.apply_along_axis(L, 1, a) * c[i]
    r4 = np.apply_along_axis(L, 1, a * c[i])
    e = np.max(np.abs(r3 - r4))
    if e > e_max : 
      return False
  return True

#V = 3
#W = 4
# def L(x):
#   return np.array([ 
#     -6*x[1] + 2*x[2],
#     x[0] - x[1] + x[2],
#     -x[0] + x[1] -6*x[2],
#     3*x[0] - x[1] + 4*x[2]
#   ])

V = 2
W = 1
def L(x):
  return np.array([ 
    x[0]*x[1]
  ])

if isLineal(L, V, W) :
  print('La transformacion es lineal')
else :
  print('La transformacion No es lineal')
  exit(0)


while(True):
  while True:
    entrada = input('Ingrese un valor de %d dimensiones, separado por comas:' % (V))
    try:
      x = [float(s) for s in entrada.split(',')]
      if len(x) != V :
        raise ValueError('Dimensiones incorrectas')
      break
    except:
      print('Valor invalido! intente de nuevo')
      print()
      pass
  print('El resultado de la transformacion es: ', L(x))
  print()

  
