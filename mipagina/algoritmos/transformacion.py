import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def limpiar_dataset(ruta_archivo):
    # Leer el archivo CSV
    data = pd.read_csv(ruta_archivo, sep=",")

    # Manejo de nulos
    if data.isnull().any().any():
        print(data.isnull().sum())

        for columnas in data.columns:
            if data[columnas].isnull().any():
                cantidad_nulos = data[columnas].isnull().sum()
                porcentaje_nulos = (cantidad_nulos / data.shape[0]) * 100

                print(f'Columna {columnas} tiene {cantidad_nulos} valores nulos de un total de {data.shape[0]} lo que equivale al {porcentaje_nulos:.1f}%')

                if porcentaje_nulos > 60:
                    print("Eliminando columna...")
                    data = data.drop(columnas, axis=1)
                    print("Columna eliminada")
                elif 10 <= porcentaje_nulos <= 59:
                    print("Remplazando por la media/promedio...")
                    if data[columnas].dtype == 'object':
                        data[columnas].fillna(data[columnas].mode()[0], inplace=True)
                    else:
                        data[columnas].fillna(data[columnas].mean(), inplace=True)
                    print("Valores nulos remplazados por la media/promedio")
                elif porcentaje_nulos < 10:
                    print("Eliminando filas con nulos...")
                    data = data.dropna(subset=[columnas])
                    print("Filas eliminadas")
                else:
                    print("No procesado")

    # cuando sea algún dato tipo objeto tengo que transformarlos a numéricos
    columnas_obj = []
    diccionario_conversion = {}

    # Bucle para recorrer las columnas
    for i, columna in enumerate(data.columns):
        if data[columna].dtypes != 'int64' and data[columna].dtypes != 'float64':
            columnas_obj.append(i)

            # Crear diccionario de conversión
            le = LabelEncoder()
            valores_originales = data[columna].unique()
            valores_convertidos = le.fit_transform(valores_originales)
            diccionario_conversion[columna] = dict(zip(valores_originales, valores_convertidos))

    x = data.iloc[::].values

    for i in columnas_obj:
        columna = data.columns[i]
        le = LabelEncoder()
        x[:, i] = le.fit_transform(x[:, i])

    data_t = pd.DataFrame(x, columns=[data.columns])

    for col in data_t.columns:
        data_t[col] = pd.to_numeric(data_t[col], errors='coerce')

    return data_t, diccionario_conversion

# Obtener la ruta absoluta del script actual
current_directory = os.path.dirname(__file__)
absolute_path = os.path.join(current_directory, 'diabetes.csv')

# Llamar a la función
dataset_limpo, diccionario_conversion = limpiar_dataset(absolute_path)

# Imprimir el dataset limpio
print(dataset_limpo)

# Imprimir el diccionario de conversión
#print(diccionario_conversion)
