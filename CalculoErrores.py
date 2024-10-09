from sklearn import metrics
import numpy as np
def errores(reales, observados):
    mse = metrics.mean_squared_error(observados, reales)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(observados, reales)
    mape = metrics.mean_absolute_percentage_error(observados, reales) * 100  # Multiplicamos por 100 para porcentaje
    r2 = metrics.r2_score(observados, reales)
    max_error = metrics.max_error(observados, reales)
    return {
        "mse":mse,
        "rmse":rmse,
        "mae":mae,
        "mape":mape,
        "r2":r2,
        "max_error":max_error
    }

def mostrar_resultados(resultados):
    print(f"Error Medio Cuadrático (MSE): {resultados['mse']:.2f}")
    print(f"Raíz del error Medio Cuadrático (RMSE): {resultados['rmse']:.2f}")
    print(f"Error Absoluto Medio (MAE): {resultados['mae']:.2f}")
    print(f"Porcentaje de Error Absoluto Medio (MAPE): {resultados['mape']:.2f}%")
    print(f"Coeficiente de Determinación (R²): {resultados['r2']:.2f}")
    print(f"Error Máximo: {resultados['max_error']:.2f}")