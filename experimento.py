import warnings
warnings.filterwarnings("ignore")

import sys
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from SEEDEDKmeans import SKmeans
from sklearn.metrics import accuracy_score, cohen_kappa_score

sca = MinMaxScaler()

caminho = 'D:/Drive UFRN/bases/'
base = 'reuters'

dados = pd.read_csv(caminho + base + '.csv')
X = sca.fit_transform(dados.drop(['classe'], axis=1).values)
Y = dados['classe'].values

rotulados = [50, 100, 150, 200, 250, 300]
porcentagem = [0.0047, 0.0093, 0.0140, 0.0186, 0.0233, 0.0279]

resultado = pd.DataFrame()
acuraciai = []
acuraciat = []
kappai = []
kappat = []

for r, p in enumerate(porcentagem):
    
   
    inicio = time.time()
    
    
    print('Teste: '+str(rotulados[r]))
    
    X_train, X_test, y_train, y_test = train_test_split(X,Y, train_size=0.9, test_size=0.1, stratify=Y)
            
    L, U, y, yu = train_test_split(X_train, y_train, train_size = p, test_size= 1.0 - p, stratify=y_train)
   
    kmeans = SKmeans(n_grupos=np.size(np.unique(y)))
    kmeans.fit(L, U, y)
           
    
    """ FASE TRANDUTIVA """
    acuraciat.append(accuracy_score(yu, kmeans.predict(U)))
    kappat.append(cohen_kappa_score(yu, kmeans.predict(U)))
    
    """ FASE INDUTIVA """
    acuraciai.append(accuracy_score(y_test, kmeans.predict(X_test)))
    kappai.append(cohen_kappa_score(y_test, kmeans.predict(X_test)))
        
        
    fim = time.time()
    tempo = np.round((fim - inicio)/60,2)
    print('........ Tempo: '+str(tempo)+' minutos.')

resultado['R'] = rotulados
resultado['AT'] = acuraciat
resultado['KT'] = kappat
resultado['KI'] = acuraciai
resultado['KI'] = kappai
                  
resultado.to_csv('D:/Drive UFRN/Doutorado/Resultados/Artigo KBS/SEEDED Kmeans/'+base+'.csv', index=False)




    