import numpy as np
import matplotlib.pyplot as plt
import csv, operator
import itertools
from scipy import special

# Se cargan los datos t,x,y del archivo csv
archivo = open('data.csv')
entrada = csv.reader(archivo)
t = np.array([])
x = np.array([])
y = np.array([])

for i in range(10000):
    t0, x0, y0 = next(entrada)
    t0 = float(t0)
    x0 = float(x0)
    y0 = float(y0)
    t = np.append(t,t0)
    x = np.append(x,x0)
    y = np.append(y,y0)

# Se generan diferentes valores para los parametros de los ajustes
A1 = np.arange(1.2,1.45,0.05)
w1 = np.arange(2.9,3.15,0.05)
w3 = np.arange(3.4,3.8,0.05)

A2 = np.arange(1.6,2.5,0.1)
w2 = np.arange(49.9,50.15,0.05)
w4 = np.arange(58.0,62.0,0.05)

# Se combinan todas las posibilidades usando el producto cartesiano
coefs1 = list(itertools.product(A1,w1,A2,w2))
coefs2 = list(itertools.product(A1,w3,A2,w4))

# Se definen las funciones que se ajustaran a los datos
def f1(z,A1,w1,A2,w2,sign):
    
    y = -A1*np.cos(w1*z)+sign*A2*np.cos(w2*z)
    return y

def f2(z,A1,w3,A2,w4,sign):
    
    y1 = special.ellipj(w3*z,0.5)
    y2 = special.ellipj(w4*z,0.5)
    y = -A1*y1[1]+sign*A2*y2[1]
    return y

# Se define una funcion para calcular chi cuadrado
def chi2(z_o,z_t):
    
    y = 0
    for i in range(len(z_o)):
        y += (z_o[i]-z_t[i])**2/abs(z_t[i])
    return y

chix1 = []
chix2 = []
chiy1 = []
chiy2 = []

# Se calcula y guarda el chi cuadrado de cada combinacion de parametros

for j in range(len(coefs1)):
    chix1.append(chi2(x,f1(t,coefs1[j][0],coefs1[j][1],coefs1[j][2],coefs1[j][3]
                           ,1)))
    chiy1.append(chi2(y,f1(t,coefs1[j][0],coefs1[j][1],coefs1[j][2],coefs1[j][3]
                           ,-1)))

for k in range(len(coefs2)):
    chix2.append(chi2(x,f2(t,coefs2[k][0],coefs2[k][1],coefs2[k][2],coefs2[k][3]
                           ,1)))
    chiy2.append(chi2(y,f2(t,coefs2[k][0],coefs2[k][1],coefs2[k][2],coefs2[k][3]
                           ,-1)))

# Se obtiene el menor chi cuadrado y los parametros correspondientes
c_mx1 = chix1.index(min(chix1))
c_mx2 = chix2.index(min(chix2))
c_my1 = chiy1.index(min(chiy1))
c_my2 = chiy2.index(min(chiy2))


'''
# Para ver los mejores parametros de las funciones
print coefs1[c_mx1]
print coefs2[c_mx2]
print coefs1[c_my1]
print coefs2[c_my2]
'''

x_t1 = f1(t,coefs1[c_mx1][0],coefs1[c_mx1][1],coefs1[c_mx1][2],coefs1[c_mx1][3]
          ,1)
x_t2 = f2(t,coefs2[c_mx2][0],coefs2[c_mx2][1],coefs2[c_mx2][2],coefs2[c_mx2][3]
          ,1)
y_t1 = f1(t,coefs1[c_my1][0],coefs1[c_my1][1],coefs1[c_my1][2],coefs1[c_my1][3]
          ,-1)
y_t2 = f2(t,coefs2[c_my2][0],coefs2[c_my2][1],coefs2[c_my2][2],coefs2[c_my2][3]
          ,-1)

chi2x1 = min(chix1)
chi2x2 = min(chix2)
chi2y1 = min(chiy1)
chi2y2 = min(chiy2)

# Se grafican los datos y los mejores ajustes segun el criterio chi cuadrado

plt.figure()
plt.plot(t,x,"b.",label='Medidas')
plt.plot(t,x_t1,'r',label='Cosenos, Chi2 = {0}'.format(chi2x1))
plt.xlabel('Tiempo')
plt.ylabel('x')
plt.grid()
plt.legend()
plt.savefig("xvst_cosenos.png")
plt.close()

plt.figure()
plt.plot(t,x,"b.",label='Medidas')
plt.plot(t,x_t2,'r',label='Jacobi, Chi2 = {0}'.format(chi2x2))
plt.xlabel('Tiempo')
plt.ylabel('x')
plt.grid()
plt.legend()
plt.savefig("xvst_jacobi.png")
plt.close()

plt.figure()
plt.plot(t,y,'b.',label='Medidas')
plt.plot(t,y_t1,'r',label='Cosenos, Chi2 = {0}'.format(chi2y1))
plt.xlabel('Tiempo')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.savefig("yvst_cosenos.png")
plt.close()

plt.figure()
plt.plot(t,y,'b.',label='Medidas')
plt.plot(t,y_t2,'r',label='Jacobi, Chi2 = {0}'.format(chi2y2))
plt.xlabel('Tiempo')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.savefig("yvst_jacobi.png")
plt.close()

# Se grafica y vs x para ver si existe alguna correlacion

plt.figure()
plt.plot(x,y,'b.',markersize=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.savefig("yvsx.png")
plt.close()
