import numpy as np
from sklearn import metrics


def errores(reales, observados):
    mse = metrics.mean_squared_error(observados, reales)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(observados, reales)
    mape = metrics.mean_absolute_percentage_error(observados, reales) * 100  # Multiplicamos por 100 para porcentaje
    r2 = metrics.r2_score(observados, reales)
    max_error = metrics.max_error(observados, reales)
    return {
        "mse ":mse,
        "rmse ":rmse,
        "mae ":mae,
        "mape ":mape,
        "r2 ":r2,
        "max_error ":max_error
    }



#Temperatura
#Descripción: Medición de la temperatura ambiente en una habitación.
Temperaturas_Reales     = [22, 22.1, 21.9, 22.2, 22.0]
Temperaturas_Observados = [ 22, 22.3, 21.7, 22.1, 21.8]

print(errores(Temperaturas_Reales, Temperaturas_Observados))

# Tiempo de Reacción
# Descripción: Medición del tiempo de reacción en un experimento de psicología.
Reacciones_Reales       = [250, 240, 260, 230, 250]
Reacciones_Observados   = [300, 220, 290, 240, 400]

print(errores(Reacciones_Reales, Reacciones_Observados))

#Distancia
# Descripción: Medición de la distancia entre dos puntos en un laboratorio.
Valores_Reales      = [5.0, 5.1, 4.9, 5.0, 5.0]
Valores_Observados  = [5.0, 5.0, 5.0, 5.0, 5.0]

print(errores(Valores_Reales, Valores_Observados))

