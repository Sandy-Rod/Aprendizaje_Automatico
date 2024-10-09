import CalculoErrores as ce
# Datos
medicion = [43, 61, 37, 35, 79, 37, 60, 38, 30, 69, 88, 82, 95, 72, 71, 30, 74, 39, 91, 36, 65, 46, 74, 31, 94, 36, 71,
            98, 89, 54]
reales = [44, 66, 38, 39, 81, 40, 62, 39, 33, 71, 88, 85, 99, 73, 71, 34, 77, 44, 94, 40, 69, 51, 76, 33, 96, 36, 75,
          99, 91, 57]
# Calculo el error
err=ce.errores(medicion,reales)
# Imprimimos los resultados
ce.mostrar_resultados(err)