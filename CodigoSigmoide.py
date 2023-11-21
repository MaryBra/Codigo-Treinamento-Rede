import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(X, W1, W2):
    bias = 1
    Xb = np.insert(X, 0, bias, axis=0)
    o1 = sigmoid(W1.dot(Xb))
    o1b = np.insert(o1, 0, bias)
    Y = sigmoid(W2.dot(o1b))
    return Y

#Carregando o arquivo .csv
dataset = pd.read_csv('diabetes.csv')

#Extrair os arrays de atributos e rótulos do conjunto de dados
atributos = dataset[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].values
rotulos = dataset['Outcome'].values

#Funcao que transforma os numeros inteiros altos em uma media pra que fiquem em um formato padrão
scaler = StandardScaler()
scaler.fit(atributos)
atributos = scaler.transform(atributos)

# Configurações da rede neural
numEpocas = 10000            #Numero de épocas
q = len(rotulos)            #Numero de padrões
eta = 0.01                 #Taxa de aprendizado 
m = atributos.shape[1]      #Numero de neuronios na camada de entrada (8 atributos)
N = 4                       #Numero de neuronios na camada escondida 
L = 1                       #Numero de neuronios na camada de saida (output)

#loop para substituir os valores 0 do dataset por -1 para se adequar melhor a função sigmoide
# for i in range(len(rotulos)):
#     if rotulos[i] == 0:
#         rotulos[i] = -1

# Inicia aleatoriamente as matrizes de pesos.
W1 = np.random.random((N, m + 1)) #Dimensoes da Matriz de entrada
W2 = np.random.random((L, N + 1)) #Dimensoes da Matriz de saida

# Array para armazenar os erros.
E = np.zeros(q)
Etm = np.zeros(numEpocas) #Etm = Erro total médio ==> serve para acompanharmos a evolução do treinamento da rede

# bias
bias = 1

# Entrada do Perceptron.
X = atributos.T

# Treinamento
for i in range(numEpocas):
    for j in range(q):
        Xb = np.insert(X[:, j], 0, bias)  # Inserir o bias diretamente no vetor de entrada
        o1 = sigmoid(W1.dot(Xb))
        o1b = np.insert(o1, 0, bias)
        Y = sigmoid(W2.dot(o1b))
        e = rotulos[j] - Y
        E[j] = (e.transpose().dot(e))/2

        # Error backpropagation.
        delta2 = np.diag(e).dot(Y*(1 - Y))  # Derivada da sigmoide aplicada aqui
        vdelta2 = (W2.transpose()).dot(delta2)
        delta1 = np.diag(o1b*(1 - o1b)).dot(vdelta2)

        # Atualização dos pesos.
        W1 = W1 + eta*(np.outer(delta1[1:], Xb))
        W2 = W2 + eta*(np.outer(delta2, o1b))

    Etm[i] = E.mean()

# Visualização do erro
plt.xlabel("Épocas")
plt.ylabel("Erro Médio")
plt.plot(Etm, color='b')
plt.show()

# Avaliação da rede no conjunto de teste
for i in range(len(rotulos)):
    exemplo_teste = atributos[i, :]
    previsao = predict(exemplo_teste, W1, W2)
    print(f"Previsão para o exemplo {i + 1}: {previsao} => {rotulos[i]}")

# Cálculo dos erros
Error_Test = np.zeros(q)
for i in range(q):
    Xb = np.hstack((bias, X[:,i]))

    o1 = sigmoid(W1.dot(Xb))     
    o1b = np.insert(o1, 0, bias)
    Y = sigmoid(W2.dot(o1b))

    Error_Test[i] = rotulos[i] - (Y)

print("Erros: " + str(np.round(Error_Test) - rotulos))
