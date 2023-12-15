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
from sklearn.neighbors import KNeighborsClassifier
test_scores = []
train_scores = []
for i in range(1,15):
    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)
    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))
max_train_score = max(train_scores)
train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))    

max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
best_k = test_scores_ind[0] + 1  # El mejor k es el primero en la lista de índices

print('Max test score {} % and k = {}'.format(max_test_score * 100, list(map(lambda x: x + 1, test_scores_ind))))

nn = KNeighborsClassifier(n_neighbors=best_k)

# Utiliza nn en lugar de knn para ajustar y evaluar el modelo
nn.fit(X_train, y_train)
print(nn.score(X_test, y_test))