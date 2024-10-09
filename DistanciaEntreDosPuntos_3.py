import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn import metrics

# Definimos las coordenadas de las tiendas
puntos = {
    'Punto A': (2, 3),
    'Punto B': (5, 4),
    'Punto C': (1, 1),
    'Punto D': (6, 7),
    'Punto E': (3, 5),
    'Punto F': (8, 2),
    'Punto G': (4, 6),
    'Punto H': (2, 1)
}
 
# Convertimos las coordenadas a un DataFrame para facilitar el cálculo
df_puntos = pd.DataFrame(puntos).T
df_puntos.columns = ['X', 'Y']
print("Coordenadas de las tiendas:")
print(df_puntos)

# Inicializamos un DataFrame para almacenar las distancias
distancias_eu = pd.DataFrame(index=df_puntos.index, columns=df_puntos.index)
distancias_mh = pd.DataFrame(index=df_puntos.index, columns=df_puntos.index)
distancias_ch = pd.DataFrame(index=df_puntos.index, columns=df_puntos.index)


# Cálculo de distancias
for i in df_puntos.index:
    for j in df_puntos.index:
        if (i != j):
            # Distancia Euclidiana
            distancias_eu.loc[i, j] = distance.euclidean(df_puntos.loc[i], df_puntos.loc[j])
 
            # Distancia Manhattan
            distancias_mh.loc[i, j] = distance.cityblock(df_puntos.loc[i], df_puntos.loc[j])
 
            # Distancia Chebyshev
            distancias_ch.loc[i, j] = distance.chebyshev(df_puntos.loc[i], df_puntos.loc[j])


 # Mostrar los resultados
print("\nDistancias Euclidianas entre las tiendas:")
print(distancias_eu)
 
print("\nDistancias Manhattan entre las tiendas:")
print(distancias_mh)
 
print("\nDistancias Chebyshev entre las tiendas:")
print(distancias_ch)

#¿Cuales serán los puntos más cercanos y cuales los más alejados dependiendo de la distancia?
# Distancia más corta
print("\n Distancia más corta Euclidianas: ",  distancias_eu.min().min())
print("\n Distancia más corta Manhattan: ",    distancias_mh.min().min())
print("\n Distancia más corta Chebyshev: ",    distancias_ch.min().min())

# Distancia más larga
print("\n Distancia más larga Euclidianas: ",  distancias_eu.max().max())
print("\n Distancia más larga Manhattan: ",    distancias_mh.max().max())
print("\n Distancia más larga Chebyshev: ",    distancias_ch.max().max())


