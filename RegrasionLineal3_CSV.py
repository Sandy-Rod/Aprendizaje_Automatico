from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt

# Cargar el conjunto de datos "diabetes"
diabetes = load_diabetes()

# Crear un DataFrame con los datos
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
print(diabetes)

#Añadimos la columna de progreso de la diabetes
df['Progression'] = diabetes.target

# Mostrar las primeras filas del DataFrame
print(df.head())

# Dividir los datos en variables predictoras (X) y la variable objetivo (y)
X = df.drop(columns=['Progression'])
y = df['Progression']

# Dividir el dataset en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Mostrar los tamaños de los conjuntos
print("\n Tamaño del conjunto de entrenamiento:", X_train.shape)
print("\n Tamaño del conjunto de prueba:", X_test.shape)

# Crear el modelo de regresión lineal
model = LinearRegression()


# Entreno el modelo con los datos que yo he seleccionado para el entrenamiento
model.fit(X_train, y_train)


# Realizar predicciones
y_pred = model.predict(X_test)




# Información del modelo
# Obtener los coeficientes e intercepto
print("\nCoeficiente:", model.coef_)
print("\nIntercepto:", model.intercept_)


# Calcular métricas
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Imprimir resultados
print(f"\nError Cuadrático Medio (MSE): {mse:.2f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.2f}")
print(f"Error Absoluto Medio (MAE): {mae:.2f}")
print(f"Coeficiente de Determinación (R²): {r2:.2f}")
