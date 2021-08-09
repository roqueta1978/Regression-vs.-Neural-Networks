# https://towardsdatascience.com/linear-regression-v-s-neural-networks-cd03b29386d4

'''El trabajo está basado en el conjunto de datos de Eficiencia Energética de la UCI.
En el contexto de los datos, se está trabajando con cada columna que se define de la siguiente manera:
X1 - Compacidad relativa
X2 - Área de superficie
X3 - Área de la pared
X4 - Área del techo
X5 - Altura total
X6 - Orientación
X7 - Área de acristalamiento
X8 - Distribución del área de acristalamiento
y1 - Carga de calefacción
y2 - Carga de enfriamiento
El principal objetivo es predecir la carga de calefacción y refrigeración en función del X1-X8.'''


# Librerías
## Tratamiento de datos
import pandas as pd
## Funciones algebráicas
import numpy as np
## Visualización
import matplotlib.pyplot as plt
import seaborn as sns
## Machine Learning y preprocesamiento de datos
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
### Eliminación recursiva de características
from sklearn.feature_selection import RFE
### Análisis de componentes principales para reducción de la dimensionalidad
from sklearn.decomposition import PCA
## Métricas
from sklearn.metrics import r2_score, mean_squared_error
## Redes Neuronales
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
### La optimización de Adam es un método de descenso de gradiente estocástico
### que se basa en la estimación adaptativa de momentos de primer y segundo orden.
from tensorflow.keras.optimizers import Adam
### Detiene el entrenamiento cuando una métrica monitoreada deja de mejorar
from tensorflow.keras.callbacks import EarlyStopping
## Modelo de mínimos cuadrados ordinarios
import statsmodels.api as sm
## Factor de inflación de la varianza
from statsmodels.stats.outliers_influence import variance_inflation_factor

## Eliminación de avisos
import warnings
warnings.filterwarnings('ignore')

# Carga del set de datos

df=pd.read_excel('ENB2012_data.xlsx')
df.head()
df.info()


df.describe()

'''No hay valores nulos'''


# Comprobación de la distribución de los datos en cada una de las variables

variables=df.columns
variables


## boxplot de cada una de las variables numéricas

plt.figure(figsize=(15,7))
for i, col in enumerate(variables):
    plt.subplot(2,5,i+1)
    sns.boxplot(y=col, data=df, color='peru')
    plt.xlabel(col, weight='bold')
    plt.ylabel('Valores', weight='bold')
plt.tight_layout(pad=1.1)
plt.show()


'''La distribución de los datos en las variables parece ajustarse en la
mayoría de los casos a una distribución normal'''

## Histograma de distribución de cada una de las variables numéricas
plt.figure(figsize=(15,7))
for i, col in enumerate(variables):
    plt.subplot(2,5,i+1)
    sns.histplot(df[col], color='blue', stat='frequency')
    plt.xlabel(col, weight='bold')
    plt.ylabel('Frecuencia', weight='bold')
plt.tight_layout(pad=1.1)
plt.show()
'''A través de los histogramas, se observa que ninguna de las variables se ajusta a una distribución normal.'''

## Multicolinealidad

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='rainbow')
plt.show()


'''Se observa perfectamente que puedan existir problemas de Multicolinealidad entre
algunas de las variables independientes. Por ejemplo, X1 con X2, X4 con X2. Esta claro que utilizando
un modelo de regresión sería conveniente reducir la dimensionalidad o el número de características.'''

## Comprobación de las variables que presentan Multicolinealidad

X=df.drop(columns=['Y1', 'Y2'])

y1=df[['Y1']]
y2=df[['Y2']]


X=sm.add_constant(X)

X

model=sm.OLS(y1,X)
result1=model.fit()
print(result1.summary())

model=sm.OLS(y2,X)
result2=model.fit()
print(result2.summary())

'''La variable X6 no resulta significativa para el modelo cuando la variable objeto es y1.
Cuando la variable objeto es y2 no resultan explicativas del modelo las variable X6 y X8'''


## Comprobación de la multicolinealidad a traves del factor de influencia de la varianza

vif=pd.DataFrame()
vif['variables']=X.columns
vif['VIF']=[variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif


'''Los valores infinitos mostrados en las variables X1, X2 y X3 indican que existe una correlación perfecta, por lo que
es indudable la existencia de multicolinealidad.'''


# Selección de las características más relevantes del set de datos.


lr=LinearRegression()

modelo=RFE(estimator=lr, n_features_to_select=4)


modelo.fit(X, y1)


print(modelo.ranking_)

print(modelo.support_)

'''Como se puede observar las varialbes de mayor interés son las que muestran un mayor grado de multicolinealidad'''

'''La mejor solución es reducir la dimensionalidad e intentar que se pierda la menor información posible'''

# Reducción de la dimensionalidad (PCA):

pca=PCA(n_components=8)

pca.fit(X)

variance=pca.explained_variance_ratio_*100

variance[0]+variance[1]


'''Se puede observar que los dos primeros componentes principales explican más del 90% de la varianza'''


pca=PCA(n_components=2)
X=pca.fit_transform(X)

X
X=pd.DataFrame(X, columns=['PC_1', 'PC_2'])

X
y1
y2


# Modelo de regresión líneal

plt.figure(figsize=(8,8))
sns.heatmap(pd.concat([X, y1, y2], axis=1).corr(), annot=True,
fmt='.2f', cmap='viridis')
plt.show()

'''Se puede observar perfectamente que no existe correlación entre las variables independientes, pero
sí entre éstas y las variables objeto'''

plt.figure(figsize=(10,7))
for i, col in enumerate(X.columns):
    plt.subplot(1,2,i+1)
    sns.histplot(X[col], kde=True, stat='frequency', color='blue')
    plt.xlabel(col, weight='bold')
    plt.ylabel('Frecuencia', weight='bold')
plt.tight_layout(pad=1.1)
plt.show()

X.describe()


'''Sigue sin observarse una distribución normal'''

## Normalización de los datos (escala estándar)

scale=StandardScaler()
X=scale.fit_transform(X)

X=pd.DataFrame(X, columns=['PC_1', 'PC_2'])
X

X.describe()

Y=pd.concat([y1, y2], axis=1)

## Separación de datos en entrenamiento y evaluación

X_train, X_test, Y_train, Y_test=train_test_split(X, Y, random_state=42, test_size=0.25)


## Modelado

model=LinearRegression()

model.fit(X_train, Y_train)

Y_pred=model.predict(X_test)

Y_pred

print(f'R2_score={r2_score(Y_test, Y_pred)}')
print(f'MSE={mean_squared_error(Y_test, Y_pred)}')
print(f'RMSE={mean_squared_error(Y_test, Y_pred, squared=False)}')

Y_test['Y1']


Y_pred[:, 0]

### Y1_Real vs. Y1_Predicha
plt.figure(figsize=(10,8))
plt.scatter(Y_test['Y1'], Y_pred[:,0], color='red')
plt.xlabel('Y1_Real', weight='bold')
plt.ylabel('Y1_Predicha', weight='bold')
plt.show()


### Y2_Real vs. Y2_Predicha
plt.figure(figsize=(10,8))
plt.scatter(Y_test['Y2'], Y_pred[:,1], color='red')
plt.xlabel('Y2_Real', weight='bold')
plt.ylabel('Y2_Predicha', weight='bold')
plt.show()


# Modelo de Red Neuronal:

## Separación de datos
### entrenamiento

X_train_red=X_train[:401]
X_val_red=X_train[401:]

Y_train_red=Y_train[:401]
Y_val_red=Y_train[401:]


## Secuencial
model=Sequential()
## Primera capa
model.add(Dense(28, input_shape=(2,), activation='relu'))
## Segunda capa
model.add(Dense(40, activation='relu'))
## Tercera capa
model.add(Dense(40, activation='relu'))
## Cuarta capa
model.add(Dense(28, activation='relu'))
## Quinta capa (salida)
model.add(Dense(2, activation='relu'))

## Compilación
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001, decay=0.001/200), metrics=['mse'])

## Entrenamiento
es=EarlyStopping(monitor='val_loss', patience=200, mode='min', verbose=1)
history=model.fit(X_train_red, Y_train_red,
          validation_data=(X_val_red, Y_val_red),
          callbacks=[es],
          epochs=10000,
          batch_size=100)
summary=history.history
summary


### Entrenamiento vs. Validación
epochs=range(len(summary['val_loss']))
epochs

plt.figure(figsize=(10,8))
plt.plot(epochs, summary['loss'], '-.', label='Entrenamiento')
plt.plot(epochs, summary['val_loss'],'-', label='Validación')
plt.xlabel('Nº de iteraciones', weight='bold')
plt.ylabel('Pérdida', weight='bold')
plt.legend()
plt.show()

## Cálculo del r2_score

Y_pred=model.predict(X_test)

print(f'R2_score={r2_score(Y_test, Y_pred)}')
print(f'MSE={mean_squared_error(Y_test, Y_pred)}')
print(f'RMSE={mean_squared_error(Y_test, Y_pred, squared=False)}')


'''Como se puede observar, se ha mejorado significativamente la precisión del modelo
mediante la utilización de la red neuronal. Además, tal y como se puede observar en el gráfico de arriba,
no se observa overfitting, por lo que el modelo está perfectamente ajustado.'''


### Y_Real vs. Y_Predicha
plt.figure(figsize=(10,8))
plt.scatter(Y_test, Y_pred, color='red')
plt.xlabel('Y_Real', weight='bold')
plt.ylabel('Y_Predicha', weight='bold')
plt.show()

### Y1_Real vs. Y1_Predicha
plt.figure(figsize=(10,8))
plt.scatter(Y_test['Y1'], Y_pred[:,0], color='blue')
plt.xlabel('Y1_Real', weight='bold')
plt.ylabel('Y1_Predicha', weight='bold')
plt.show()

### Y2_Real vs. Y2_Predicha
plt.figure(figsize=(10,8))
plt.scatter(Y_test['Y2'], Y_pred[:,1], color='peru')
plt.xlabel('Y2_Real', weight='bold')
plt.ylabel('Y2_Predicha', weight='bold')
plt.show()
