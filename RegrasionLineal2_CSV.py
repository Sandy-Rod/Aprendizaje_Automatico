from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing

# Cargar el conjunto de datos "California Housing"
california = fetch_california_housing()
print(california)
# Crear un DataFrame con los datos
df = pd.DataFrame(california.data, columns=california.feature_names)

# Añadir la columna de los precios de las casas
df['PRICE'] = california.target

# Mostrar las primeras filas del DataFrame
print(df.head())

# Dividir los datos en variables predictoras (X) y la variable objetivo (y)
X = df.drop(columns=['PRICE'])
y = df['PRICE']

# Dividir el conjunto de datos en entrenamiento y prueba (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de regresión lineal
modelo = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
modelo.fit(X_train, y_train)

# Predecir los precios con los datos de prueba
y_pred = modelo.predict(X_test)


# Calcular métricas
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Imprimir resultados
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.2f}")
print(f"Error Absoluto Medio (MAE): {mae:.2f}")
print(f"Coeficiente de Determinación (R²): {r2:.2f}")