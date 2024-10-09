import numpy as np
from sklearn import metrics

#Supongamos que tienes tres conjuntos de mediciones de la longitud de un objeto (por ejemplo, una regla).
# Cada conjunto representa un intento de medir la longitud en centímetros. Los valores reales de longitud son los siguientes:

valor_real = [50] * 5

#Conjuntos de Medidas
medicion_A = [49.8, 50.1, 50.0, 49.9, 50.2]
medicion_B = [48.0, 52.0, 49.0, 51.0, 47.0]
medicion_C = [50.5, 49.5, 50.0, 50.1, 50.3]


#Objetivo del Ejercicio
#Calcular el Error Absoluto Medio (MAE) para cada conjunto de medidas respecto a los valores reales.


MAE_A = metrics.mean_absolute_error(valor_real, medicion_A)

MAE_B = metrics.mean_absolute_error(valor_real, medicion_B)

MAE_C = metrics.mean_absolute_error(valor_real, medicion_C)


#Determinar cuál de los conjuntos de mediciones es el más preciso, es decir, el que tiene el MAE más bajo.

print(f"Error Absoluto Medio (MAE) - medición A: {MAE_A:.2f}")
print(f"Error Absoluto Medio (MAE) - medición B: {MAE_B:.2f}")
print(f"Error Absoluto Medio (MAE) - medición C: {MAE_C:.2f}")