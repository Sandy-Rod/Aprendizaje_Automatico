import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Parámetros del ejercicio
muestras = 100
dependencia = 4
ruido = 1

# Generar datos simulados
np.random.seed(42)
# Genero una serie de datos completamente aleatoria entre 0 y 2
X = 2 * np.random.rand(muestras, 1)  # 100 muestras de una variable independiente
# Relación lineal con ruido. Cuanta más dependencia, más enlazados están los datos con
# la variable independiente. Cuanto más ruido, más alejados
y = 3 + (dependencia * X) + (np.random.randn(muestras, 1) * ruido)

# Tengo una serie de datos X independientes
# Tengo una serie de datos y que depende de X pero no completamente, porque tiene una parte de ruido

# Creamos un conjunto de entrenamiento y uno de prueba

porcentaje = 0.2
tam = int(len(X) * (1-porcentaje)) # En el caso de 100 muestras el tamaño de entrenamiento es 80
X_train = X[:tam] # todos los elementos desde el 0 hasta el 80
X_test = X[tam:] # todos los elementos desde el 80 hasta el final
y_train = y[:tam]
y_test = y[tam:]

# Lo mismo usando train_test
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=porcentaje)

# Crear el modelo de regresión lineal
model = LinearRegression()
# Entreno el modelo con los datos que yo he seleccionado para el entrenamiento
# En nuestro ejemplo 80 datos
model.fit(X_train, y_train)

# Realizar predicciones
# Si mi modelo cogiera como valores los datos que he apartado de test
# ¿Qué valores me devolvería?
y_pred = model.predict(X_test)

# ¿Qué tengo ahora? Tengo, por un lado, los datos reales que son y_test
# Por otro lado los datos que me predice el modelo que son y_pred
# x_text:[0.2,1.3,0.7] y_test [8,12,9]
# y_pred: [8.1,11.9,9.2]

plt.scatter(X, y, color='blue', label="Datos Reales")
plt.plot(X_test, y_pred, color='red', label="Regresión Lineal")
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Información del modelo
# Obtener los coeficientes e intercepto
print("Coeficiente:", model.coef_)
print("Intercepto:", model.intercept_)

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