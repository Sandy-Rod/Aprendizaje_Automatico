import numpy as np
import pandas as pd
import scipy.stats as stats


#Función para calcular las estadisticas y devolver un DataFrame

def calcular_estadistica(df):
    estadisticas = pd.DataFrame(index=df.columns, columns=['Rango', 'Media', 'Mediana', 'Moda', 'Desviación estándar', 'Asimetría', 'Curtosis'])
    
    for col in df.columns:
        rango           = df[col].max() - df[col].min()  # Calcular el rango
        media           = df[col].mean()  # Media
        mediana         = df[col].median()  # Mediana
        moda            = df[col].mode()[0]  # Moda (primer valor si hay más de una moda)
        desviacion_std  = df[col].std()  # Desviación estándar
        asimetria       = stats.skew(df[col])  # Asimetría
        kurtosis        = stats.kurtosis(df[col])  # Curtosis
        
        # Llenar el DataFrame con los resultados
        estadisticas.loc[col] = [rango, media, mediana, moda, desviacion_std, asimetria, kurtosis]
    
    return estadisticas 


# Función que calcula el coeficiente de correlación de Pearson y el p-valor de algunas columnas 
def calcular_corralacion(df, caracteristicas):
    resultados = {}
    
    for i in range(len(caracteristicas)):
        for j in range(i + 1, len(caracteristicas)):
            # Calcular correlación y p-valor
            coef_pearson, p_valor = stats.pearsonr(df[caracteristicas[i]], df[caracteristicas[j]])
            resultados[f'{caracteristicas[i]} vs {caracteristicas[j]}'] = {'Coeficiente de correlación': coef_pearson, 'P-valor': p_valor}
    # pasamos reulstado(diccionario) a dataframe 
    return pd.DataFrame(resultados).T #transponemos para que las claves del diccionario pasen a columnas y los valores a filas





