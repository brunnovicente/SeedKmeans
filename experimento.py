import warnings
warnings.filterwarnings("ignore")

import sys
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from SEEDEDKmeans import SKmeans

sca = MinMaxScaler()

dados = pd.read_csv('d:/basedados/agricultura.csv')
X = sca.fit_transform(dados.drop(['classe'], axis=1).values)
Y = dados['classe'].values

X_train, X_test, y_train, y_test = train_test_split(X,Y, train_size=0.9, test_size=0.1, stratify=Y)

rotulados = [50 , 100, 150, 200, 250, 300]
porcentagem = [0.0047, 0.0093, 0.0140, 0.0186, 0.0233, 0.0279]

rotulados = [50, 100, 150, 200, 250, 300]
porcentagem = [0.0047, 0.0093, 0.0140, 0.0186, 0.0233, 0.0279]

for r, p in enumerate(porcentagem):
    
    resultado = pd.DataFrame()
    resultadoT = pd.DataFrame()
    
    inicio = time.time()
    
    for k in np.arange(10):
        print('Teste: '+str(rotulados[r])+' - '+str(k+1))
        
        X_train, X_test, y_train, y_test = train_test_split(X,Y, train_size=0.9, test_size=0.1, stratify=Y)
                
        """ PROCESSO TRANSDUTIVO """
        L, U, y, yu = train_test_split(X_train, y_train, train_size = p, test_size= 1.0 - p, stratify=y_train)
       
        kmeans = SKmeans(n_grupos=6)
        kmeans.fit(L, U, y)
               
        
        """ PROCESSO TRANDUTIVO """
        resultado['exe'+str(k+1)] = kmeans.predict(U)
        resultado['y'+str(k+1)] = yu
        
        
        """ PROCESSO INDUTIVO """
        resultadoT['exe'+str(k+1)] = kmeans.predict(X_test)
        resultadoT['y'+str(k+1)] = y_test
        
        
    fim = time.time()
    tempo = np.round((fim - inicio)/60,2)
    print('........ Tempo: '+str(tempo)+' minutos.')
                    
    resultado.to_csv('resultados/resultado_KMEANS_'+str(rotulados[r])+'.csv', index=False)
    resultado.to_csv('resultados/resultado_KMEANS_'+str(rotulados[r])+'.csv', index=False)
    