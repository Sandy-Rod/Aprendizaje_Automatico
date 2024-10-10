import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_wine
import Utilities as utilities

# Cargar el conjunto de datos "wine"
wine=load_wine()

# Crear un DataFrame con los datos
df = pd.DataFrame(wine.data, columns=wine.feature_names)

#¿Cuántas muestras tenemos? 
print("Nº de muestras: ", df.shape[0])

# #¿Qué características se analizan?
print("Características que se analizan:", list(df.columns))

#Para cada una de las características me interesaría saber sus medidas de 
#estadística descriptiva: Rango, media, mediana, moda, desviación estándar,
#asimetría, kurtosis…
estadisticas_descriptivas =  utilities.calcular_estadistica(df)
print("\n Medidas de estadística descriptiva: \n ", estadisticas_descriptivas)


#Vamos a analizar si hay correlación entre alguna de las características. 
# Puede ser negativa, positiva o neutra. No hace falta hacerlo para todas, 
# pero en las tres más significativas podemos calcular el coeficiente p.

# Selecciono tres características: alcohol, malic_acid y flavanoids
caracteristicas = ['alcohol', 'malic_acid', 'flavanoids']

# Calculo correlaciones entre las tres características seleccionadas
print("\n Calculo correlaciones entre las tres características seleccionadas: \n")
correlaciones = utilities.calcular_corralacion(df, caracteristicas)
print(correlaciones)