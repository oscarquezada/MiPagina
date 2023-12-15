from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from transformacion import limpiar_dataset
import os
import pandas as pd
from sklearn.model_selection import train_test_split

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

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=39, stratify=y)
from sklearn.naive_bayes import GaussianNB
modeloNB = GaussianNB()
# Entrenamiento del modelo
modeloNB.fit(X_train, y_train)
# Validación del modelo
y_pred = modeloNB.predict(X_test)
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print('Precisión del modelo:', precision)