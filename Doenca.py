import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras import Sequential
from tensorflow.keras.layers import Dense # type: ignore
from sklearn.metrics import accuracy_score




base = pd.read_csv(r'C:\\Users\henri\\GitPython\\Estudos_Python\Estudos_Python\\Rede Neurais\\soybean.csv')
print(base.head())

atributos = base.iloc[:,0:35].values
classe = base.iloc[:,35].values

le = LabelEncoder()

#Padronizando os dados com o labelencoder
for i in range(35):
    atributos[:,i] = le.fit_transform(atributos[:,i]
)

#Utilizando o onehotencoder para remodelar a classe
classe = np.array(classe).reshape(-1,1)
oe = OneHotEncoder(categories='auto', sparse_output=False)
classe = oe.fit_transform(classe)

x_treino, x_teste, y_treino, y_teste = train_test_split(atributos, classe, test_size=0.3, random_state=0)
print('Dados separados entre treino e teste')

sc = StandardScaler()
x_treino = sc.fit_transform(x_treino)
x_teste = sc.fit_transform(x_teste)

modelo = Sequential()
modelo.add(Dense(units = 27, kernel_initializer = 'uniform', activation = 'relu', input_dim = 35))
modelo.add(Dense(units = 27, kernel_initializer = 'uniform', activation = 'relu'))
modelo.add(Dense(units = 19, kernel_initializer = 'uniform', activation = 'sigmoid'))
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print('===================================================')
print('Modelo criado')

modelo.fit(x_treino, y_treino, batch_size=5, epochs=150)
print('====================================================')
print('modelo treinado')

y_previsao = modelo.predict(x_teste)
print(f'Previsao: {y_previsao}')

#Transformando as variaveis para poder ser lidas
y_teste = np.argmax(y_teste, axis=1)
y_previsao = np.argmax(y_previsao, axis=1)

accuracy = accuracy_score(y_teste, y_previsao)
print(f'A taxa de acertp do modelo é igual a {accuracy}')