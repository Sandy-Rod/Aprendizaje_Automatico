import numpy as np

# valores reales
y_real = np.array([10, 20, 30, 40, 50])

# Valores observados/predicciones
y_pred = np.array([8, 22, 28, 45, 55])


# Calcular el error absoluto
error_absoluto = np.abs(y_real - y_pred)
print("Error Absoluto:", error_absoluto)

# Calcular el error cuadrático
error_cuadratico = (y_real - y_pred) ** 2
print("Error Cuadrático:", error_cuadratico)


# Calcular el Error Cuadrático Medio (MSE) 
mse = np.mean((y_real - y_pred) ** 2)
print("Error Cuadrático Medio (MSE):", mse)


# Calcular el Error Absoluto Medio (MAE)
mae = np.mean(np.abs(y_real - y_pred))
print("Error Absoluto Medio (MAE):", mae)


# Calcular el error relativo
error_relativo = np.abs(y_real - y_pred) / np.abs(y_real) * 100
print("Error Relativo (%):", error_relativo)


# R² (Coeficiente de Determinación)
ss_res = np.sum((y_real - y_pred) ** 2)
ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
r2 = 1 - (ss_res / ss_tot)
print("Coeficiente de Determinación (R²):", r2)

