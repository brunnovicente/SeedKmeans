from seedsKmeans import SEEDS
import pandas as pd 
import BaseDados as BD
import numpy as np

tamanhos= [0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

resultado = []

for tam in tamanhos:
    X,Y = BD.base_mnist_fashion('D:/basedados/')
    dados = pd.DataFrame(X, columns=np.arange(np.size(X, axis=1)))
    dados['classe'] = Y+1
    acuracia = []
    recall = []
    precisao = []
    fscore = []
    
    for j in range(0, 10):
        print('Tamanho:',tam,' - Iteração ', j)
        kmeans = SEEDS(dados, 10)
        acuracia.append(kmeans.acuracia)
        recall.append(kmeans.recall)
        precisao.append(kmeans.precisao)
        fscore.append(kmeans.f1)
    resultado.append([np.mean(acuracia), np.std(acuracia), np.mean(recall), np.mean(precisao), np.mean(fscore)])

colunas = ['ACURACIA', 'STD', 'RECALL', 'PRECISAO', 'F-SCORE']
dados = pd.DataFrame(resultado, columns=colunas)
dados.to_csv('resultado_seed_fashion.csv', index=False)