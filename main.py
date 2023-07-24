import pandas as pd
import plotly.express as px

data = pd.read_csv("ClientesBanco.csv", encoding="latin1")
data = data.dropna()
X = data.drop(["Categoria","Sexo","Categoria Cartão","Educação","Estado Civil","Faixa Salarial Anual","Mudanças Transacoes_Q4_Q1","Mudança Qtde Transações_Q4_Q1"], axis=1)
y = data.Categoria
from sklearn.neighbors import KNeighborsClassifier
#considera o n de vizinhos mais proximo
knn = KNeighborsClassifier(n_neighbors=3)
#funcao de treinamento que recebe o input e output
knn.fit(X,y)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#cria variaveis para treino e teste usando 2/3 dos dados para treino
X_train , X_test, y_train, y_test = train_test_split(X, y,train_size=2/3)
#print(X_test)
#recebe o input e output apos treinado para melhor accuracy
knn2 = KNeighborsClassifier(n_neighbors=3)
knn2.fit(X_train,y_train)
tabela_vdd = knn2.predict(X_test)
print(accuracy_score(y_test, knn2.predict(X_test)))
#print(tabela_vdd)
tabela_vdd1=pd.DataFrame(tabela_vdd, X_test["CLIENTNUM"])
#print(tabela_vdd1)
tabela_vdd1 = tabela_vdd1.rename({0:"Categoria"},axis=1)
tabela_vdd1.to_csv("dados.csv")

'''
este for mostra as tabelas
for coluna in X_test:
    grafico = px.histogram(data,x=coluna, color="Categoria")
    grafico.show()

 base de dados original
https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers
'''