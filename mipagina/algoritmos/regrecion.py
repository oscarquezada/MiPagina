from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import pandas as pd
from transformacion import limpiar_dataset

# Obtener la ruta absoluta del script actual
current_directory = os.path.dirname(__file__)
absolute_path = os.path.join(current_directory, 'tested.csv')
df, diccionario_conversion = limpiar_dataset(absolute_path)

# Especifica el índice de la columna que deseas excluir (en este caso, la segunda columna)
indice_columna_a_excluir = 1

# Selecciona todas las columnas excepto la columna especificada por índice
X = df.drop(df.columns[indice_columna_a_excluir], axis=1)

# Estandariza las características
sc_X = StandardScaler()
X = pd.DataFrame(sc_X.fit_transform(X),
                 columns=X.columns)  # Mantén los nombres de las columnas originales

# Ahora, X contiene todas las columnas excepto la columna en la posición especificada
y = df.iloc[:, indice_columna_a_excluir]

# Calcular correlaciones con la columna de salida
correlations = X.corrwith(y)

# Filtrar correlaciones significativas
significant_correlations = correlations[(correlations > 0.7) | (correlations < -0.7)]

# Seleccionar características significativas
X_significant = X[significant_correlations.index]

# Dividir el conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=39, stratify=y)

# Calcular correlaciones con la columna de salida solo en el conjunto de entrenamiento
correlations_train = X_train.corrwith(y_train)

# Filtrar correlaciones significativas en el conjunto de entrenamiento
significant_correlations_train = correlations_train[(correlations_train > 0.7) | (correlations_train < -0.7)]

print("Número de características significativas en el conjunto de entrenamiento:", len(significant_correlations_train))

if len(significant_correlations_train) > 1:
    # Regresión lineal múltiple con características significativas
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Regresión Lineal Múltiple con características significativas")
    print(f"Mean Squared Error: {mse:.2e}")
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")
elif len(significant_correlations) == 1:
    # Regresión lineal simple con la única característica significativa
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Regresión Lineal Simple con la característica significativa")
    print(f"Mean Squared Error: {mse:.2e}")
    print(f"Coefficient: {model.coef_[0]}")
    print(f"Intercept: {model.intercept_}")
else:
    # Regresión lineal múltiple con todas las características si no hay características significativas
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Regresión Lineal Múltiple con todas las características")
    print(f"Mean Squared Error: {mse:.2e}")
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")
