import numpy as np
from scipy.spatial import distance
# Definir dos puntos en 2D
P = np.array([1, 2])
Q = np.array([4, 6])

# Calcular la distancia euclidiana
print("Euclidiana: Problemas geométricos y de clasificación (k-NN)")
distancia_euclidiana = np.linalg.norm(P - Q)
print(f"Distancia euclidiana: {distancia_euclidiana}")
print(distance.euclidean(P, Q))


# Calcular la distancia de Manhattan
print("Manhattan: Diseño de circuitos, mapas urbanos")
distancia_manhattan = np.sum(np.abs(P - Q))
print(f"Distancia de Manhattan: {distancia_manhattan}")
print(distance.cityblock(P,Q))

# Calcular la distancia de Chebyshev
print("Chebyshev: Movimiento en tableros de ajedrez, logística")
distancia_chebyshev = np.max(np.abs(P - Q))
print(f"Distancia de Chebyshev: {distancia_chebyshev}")
print(distance.chebyshev(P,Q))


# Calcular la distancia de Minkowski con p=3
print("Minkowski: Clasificación, clustering, cuando no se sabe qué métrica usar")
p = 3
distancia_minkowski = np.sum(np.abs(P - Q)**p)**(1/p)
print(f"Distancia de Minkowski (p=3): {distancia_minkowski}")
print(distance.minkowski(P,Q))