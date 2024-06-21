import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import arff

# Cargar el nuevo dataset
file_path_titanic = '../Titanic_Clustered.csv'
data_titanic_arff = arff.loadarff(file_path_titanic)
data_titanic = pd.DataFrame(data_titanic_arff[0])

# Preprocesamiento de los datos
data_titanic['pclass'] = data_titanic['pclass'].str.decode('utf-8')
data_titanic['sex'] = data_titanic['sex'].str.decode('utf-8')
data_titanic['cabin'] = data_titanic['cabin'].str.decode('utf-8')
data_titanic['embarked'] = data_titanic['embarked'].str.decode('utf-8')
data_titanic['survived'] = data_titanic['survived'].str.decode('utf-8')

# Eliminar filas con valores faltantes
data_titanic_clean = data_titanic.replace('?', np.nan).dropna()

# Convertir columnas categóricas a numéricas
data_titanic_clean['sex'] = data_titanic_clean['sex'].map({'male': 0, 'female': 1})
data_titanic_clean['embarked'] = data_titanic_clean['embarked'].map({'S': 0, 'C': 1, 'Q': 2})
data_titanic_clean['pclass'] = data_titanic_clean['pclass'].astype(int)
data_titanic_clean['survived'] = data_titanic_clean['survived'].astype(int)

# Selección de columnas numéricas para clustering
X_titanic = data_titanic_clean[['pclass', 'sex', 'age', 'sibsp', 'fare', 'embarked']]

# Estandarización de los datos
scaler = StandardScaler()
X_titanic_scaled = scaler.fit_transform(X_titanic)

# Método del codo para encontrar el número óptimo de clusters
sse_titanic = []
k_range = range(1, 11)
for k in k_range:
    kmeans_titanic = KMeans(n_clusters=k, random_state=42)
    kmeans_titanic.fit(X_titanic_scaled)
    sse_titanic.append(kmeans_titanic.inertia_)

# Gráfico del método del codo
plt.figure(figsize=(10, 6))
plt.plot(k_range, sse_titanic, marker='o')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Suma de distancias al cuadrado')
plt.title('Método del Codo para k óptimo (Datos del Titanic)')
plt.show()

# Determinar k óptimo basado en el método del codo
optimal_k_titanic = 3

# Ajuste de KMeans con el número óptimo de clusters
kmeans_titanic = KMeans(n_clusters=optimal_k_titanic, random_state=42)
clusters_titanic = kmeans_titanic.fit_predict(X_titanic_scaled)

# Agregar las etiquetas de los clusters a los datos originales
data_titanic_clean['Cluster'] = clusters_titanic

# Guardar los datos clusterizados en un nuevo archivo CSV
output_titanic_file_path = 'Titanic_Clustered.csv'
data_titanic_clean.to_csv(output_titanic_file_path, index=False)

print(f"Archivo guardado en: {output_titanic_file_path}")
