import numpy as np
import pandas as pd
from scipy import stats

# Datos
scores = [7, 8, 9, 8, 7, 8, 9, 8, 7, 8]

# Convertir a un array de numpy
scores_array = np.array(scores)

# 1. Media
mean = np.mean(scores_array)
print(f"Media: {mean}")

# 2. Mediana
median = np.median(scores_array)
print(f"Mediana: {median}")

# 3. Moda
mode = stats.mode(scores_array)[0]  # stats.mode devuelve un objeto
print(f"Moda: {mode}")
# Otra manera
most_common = np.bincount(scores_array).argmax()
print(f"Moda: {most_common }")
# 4. Rango
data_range = np.ptp(scores_array)  # Rango = máximo - mínimo
print(f"Rango: {data_range}")

# 5. Varianza
variance = np.var(scores_array, ddof=0)  # ddof=0 para población
print(f"Varianza: {variance}")

# 6. Desviación Estándar
std_deviation = np.std(scores_array, ddof=0)
print(f"Desviación Estándar: {std_deviation}")

# 7. Tabla de Frecuencia
frequency_table = pd.Series(scores).value_counts().sort_index()
print("\nTabla de Frecuencia:")
print(frequency_table)

# 8. Gráfico de Histograma
import matplotlib.pyplot as plt

plt.hist(scores_array, bins=5, edgecolor='black')
plt.title('Histograma de Puntuaciones')
plt.xlabel('Puntuaciones')
plt.ylabel('Frecuencia')
plt.show()