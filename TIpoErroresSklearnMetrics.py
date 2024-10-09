from sklearn import metrics

# Datos
medicion = [43, 61, 37, 35, 79, 37, 60, 38, 30, 69, 88, 82, 95, 72, 71, 30, 74, 39, 91, 36, 65, 46, 74, 31, 94, 36, 71, 98, 89, 54]
reales = [44, 66, 38, 39, 81, 40, 62, 39, 33, 71, 88, 85, 99, 73, 71, 34, 77, 44, 94, 40, 69, 51, 76, 33, 96, 36, 75, 99, 91, 57]

# Cálculo de los errores
mse = metrics.mean_squared_error(medicion, reales)
rmse = metrics.root_mean_squared_error(medicion, reales)
mae = metrics.mean_absolute_error(medicion, reales)
mape = metrics.mean_absolute_percentage_error(medicion, reales)
r2 = metrics.r2_score(medicion, reales)
max_error = metrics.max_error(medicion, reales)

# Imprimimos los resultados
print("Errores de la medición")
print(f"Error medio cuadrático {mse:.2f}")
print(f"Raiz del error medio cuadrático {rmse:.2f}")
print(f"Error medio absoluto {mae:.2f}")
print(f"Porcentaje error medio absoluto {mape:.2f}")
print(f"R2 {r2:.2f}")
print(f"Error máximo {max_error:.2f}")