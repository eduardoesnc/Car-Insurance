import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from tkinter import *
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.feature_selection import mutual_info_classif, RFE, RFECV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTENC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB


st.set_page_config(
    page_title="Machine Learning",
    page_icon='./assets/car-logo.png',
    layout="wide",
    menu_items={
        'About': "Desenvolvedores: "
                 "Anna Carolina Lopes Dias Coêlho Bejan, "
                 "Atlas Cipriano da Silva, "
                 "Eduardo Estevão Nunes Cavalcante, "
                 "Matheus Mota Fernandes"
    },
    initial_sidebar_state='collapsed'
)


# Leitura e tratamento dos dados
@st.cache_data
def readData():
    dataset = pd.read_csv('./data/train.csv')
    return dataset

df = readData()

def tratarDados(database):
    # Apagar coluna policy_id, já que são apenas IDs
    database = database.drop(['policy_id'], axis=1)

    # Tranformar as colunas largura, tamanho e altura em apenas uma coluna chamada volume
    database['volume'] = np.log(database.length * database.width * database.height * 1e-6)
    # database = database.drop(['length', 'width', 'height'], axis=1)

    # Normalizar policy tenure com min max normalization
    # policy_df = database['policy_tenure']
    # database['policy_tenure'] = (policy_df - policy_df.min()) / (policy_df.max() - policy_df.min())

    age_of_car_outliers = database.age_of_car > database.age_of_car.quantile(0.995)
    database = database.loc[~age_of_car_outliers]

    age_of_policyholder_outliers = database.age_of_policyholder > database.age_of_policyholder.quantile(0.995)
    database = database.loc[~age_of_policyholder_outliers]

    database = database.replace({ "No" : False , "Yes" : True })

    database['model'] = database['model'].replace({'M1': 0, 'M2': 1, 'M3': 2, 'M4': 3, 'M5': 4, 'M6': 5, 'M7': 6, 'M8': 7, 'M9': 8, 'M10': 9, 'M11': 10})
    database['model'] = database['model'].astype('int64')

    database['area_cluster'] = database['area_cluster'].replace({'C1': 0, 'C2': 1, 'C3': 2, 'C4': 3, 'C5': 4, 'C6': 5, 'C7': 6, 'C8': 7, 'C9': 8, 'C10': 9, 'C11': 10, 'C12': 11, 'C13': 12, 'C14': 13, 'C15': 14, 'C16': 15, 'C17': 16, 'C18': 17, 'C19': 18, 'C20': 19, 'C21': 20, 'C22': 21})
    database['area_cluster'] = database['area_cluster'].astype('int64')

    database['fuel_type'] = database['fuel_type'].replace({'CNG': 0, 'Diesel': 1, 'Petrol': 2,})
    database['fuel_type'] = database['fuel_type'].astype('float64')

    database['segment'] = database['segment'].replace({'A': 0, 'B1': 1, 'B2': 2, 'C1': 3, 'C2': 4, 'Utility': 5})
    database['segment'] = database['segment'].astype('float64')

    database['transmission_type'] = database['transmission_type'].replace({'Automatic': 0, 'Manual': 1})
    database['transmission_type'] = database['transmission_type'].astype('float64')

    
    database.rename(columns={'policy_tenure': 'Tempo de seguro', 'turning_radius': 'Espaço necessário para curva',
                    'age_of_car': 'Idade do carro', 'volume': 'Volume', 'population_density': 'Densidade populacional',
                    'area_cluster': 'Área do segurado', 'age_of_policyholder': 'Idade do segurado',
                    'engine_type': 'Tipo do motor', 'model': 'Modelo', 'gross_weight': 'Peso máximo',
                    'displacement': 'cilindradas (cc)', 'max_torque': 'Torque máximo', 'max_power': 'Força máxima',
                    'segment': 'Segmento', 'is_adjustable_steering': 'Volante ajustável?',
                    'cylinder': 'Quantidade de cilindros', 'is_front_fog_lights': 'Tem luz de neblina?',
                    'is_brake_assist': 'Tem assitência de freio', 'length': 'Comprimento',
                    'is_driver_seat_height_adjustable': 'Banco do motorista é ajustável?',
                    'fuel_type': 'Tipo do combustível', 'is_parking_camera': 'Tem câmera de ré',
                    'transmission_type': 'Tipo de transmissão'}, inplace=True)
    
    return database


st.title('Machine Learning')

df = tratarDados(df)

# Gráfico Mutual Information Score
# st.subheader("Mutual Information Score")
# st.markdown("""<p style="font-size: 16px;text-align: center; margin-top: 0px">
#             O Mutual Information Score é uma métrica de aprendizado de máquina que mede a dependência entre duas
#             variáveis aleatórias. No caso do dataset que estamos trabalhando, estamos usando essa métrica para 
#             avaliar a relação entre as variáveis do dataset e a coluna is_claim e buscar quais seriam as mais úteis para
#             desenvolver o Machine Learning.
#             </p>""", unsafe_allow_html=True)


# def make_mi_scores(X, y):
#     X = X.copy()
#     for colname in X.select_dtypes(["object", "category", "bool"]):
#         X[colname], _ = X[colname].factorize()
#     # All discrete features should now have integer dtypes
#     discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
#     all_mi_scores = []
#     for random_state in range(0, 5):
#         miScores = mutual_info_classif(X, y, discrete_features=discrete_features, random_state=random_state)
#         all_mi_scores.append(miScores)
#     all_mi_scores = np.mean(all_mi_scores, axis=0)
#     all_mi_scores = pd.Series(all_mi_scores, name="MI Scores", index=X.columns)
#     all_mi_scores = all_mi_scores.sort_values(ascending=False)
#     return all_mi_scores


# def plot_mi_scores(scores):
#     scores = scores.sort_values(ascending=True)
#     ticks = list(scores.index)
#     fig = px.bar(y=ticks, x=scores, labels={'x': '', 'y': ''}, height=600, width=1000)
#     st.write(fig)             

#Verifica a correlação entre todas as colunas com o is_claim junto com o gráfico
# st.title("Correlação entre as colunas com o is_claim")
# st.write(df.corr()['is_claim'])

# corr_matrix = df.corr()
# fig, ax = plt.subplots(figsize = (10,8))
# sns.heatmap(corr_matrix[["is_claim"]], annot= True, cmap = 'coolwarm', ax= ax)
# st.pyplot(fig)

# y_target = df.is_claim.astype('int16')

# mi_scores = make_mi_scores(df.drop('is_claim', axis=1), y_target)

# plt.figure(dpi=100, figsize=(8, 5))
# st.set_option('deprecation.showPyplotGlobalUse', False)
# plot_mi_scores(mi_scores.head(20))

# -----------------------------------------------------XGBoost Classifier----------------------------------------------------- #
# st.markdown("---")
# st.subheader("XGBoost Classifier")

# Volume, Tempo de seguro, Idade do carro, Área do segurado, Densidade populacional, Idade do segurado, Modelo
# colsSelecionadasXGB = ['Volume', 'Tempo de seguro', 'Idade do carro', 'Área do segurado', 'Densidade populacional', 'Idade do segurado', 'Modelo']

# tempDF = df[colsSelecionadasXGB]

# categorical_cols = tempDF.select_dtypes(include=['object']).columns

# dfXGB = pd.get_dummies(tempDF, columns=categorical_cols)

# x = dfXGB

# y = df['is_claim']

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# smt = SMOTEENN()
# x_res, y_res = smt.fit_resample(x, y)

# modelo = XGBClassifier()

# # Print para ver os parâmetros utilizados
# # print(modelo)

# modelo.fit(x_res, y_res)

# preds = modelo.predict(x_test)

# y_test = y_test.astype(int)

# print(classification_report(y_test, preds))

# cols = x.columns
# importance = modelo.feature_importances_
# feature_importance = pd.DataFrame({'Feature': cols, 'Importance': importance})
# feature_importance = feature_importance.sort_values('Importance', ascending=False).reset_index(drop=True)
# print(feature_importance)

# conf_matrix = confusion_matrix(y_test, preds)
# print(conf_matrix)

# ConfusionMatrixDisplay.from_predictions(y_test, preds, normalize='true')

# st.caption('O resultado foi horrível')


# -----------------------------------------------------RANDOM FOREST----------------------------------------------------- #
st.markdown("---")
st.subheader("Random Forest")
st.markdown("""<p style="font-size: 16px;">
           Features selecionadas: Comprimento, Tempo de seguro, Idade do carro, Área do segurado, Idade do segurado, Modelo.
           </p>""", unsafe_allow_html=True)

if st.checkbox('Mostrar código'):
   st.code("""
# Selecionando as colunas Comprimento, Tempo de seguro, Idade do carro, Área do segurado, Idade do segurado, Modelo.
colsSelecionadasRF = ['Comprimento', 'Tempo de seguro', 'Idade do carro', 'Idade do segurado','Área do segurado', 'Modelo']

# Criando dataframe temporário apenas com as colunas selecionadas
dfRF = df

# Selecionando as colunas Categóricas
categorical_cols = dfRF.select_dtypes(include=['object']).columns

# Convertendo as colunas categóricas em variáveis de indicação
dfRF = pd.get_dummies(dfRF, columns=categorical_cols)

# Mostrar as variáveis de indicação criadas
dfRF.info(verbose = True)

# Atribuindo as colunas selecionadas a X
# x = dfRF
x = dfRF.drop(['is_claim'], axis = 1)

# Atribuindo a coluna alvo a Y
y = df["is_claim"]

# Fazendo a reamostragem para equilibrar os valores de Y
smt = SMOTEENN()
X_res, y_res = smt.fit_resample(x, y)
X_res = X_res[colsSelecionadasRF]

y_res.value_counts()

# Dividindo o dataset para treino e para teste, utilizando 20% do dataset para teste
x_train, x_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.20)

# Inicializando o algoritmo Random Forest
classifier = RandomForestClassifier()
# classifier = XGBClassifier()
 
#  Construindo uma 'floreste de árvores' da parte de treino
classifier.fit(x_train,y_train)

# Realizando as previsões
preds = classifier.predict(x_test)

# Criando o relatório com as principais métricas da classificação
report = classification_report(y_test, preds, output_dict=True)
df_report = pd.DataFrame(report).transpose()

st.table(df_report.style.format({'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}', 'support': '{:.2f}'}))
""")

# Selecionando as colunas Comprimento, Tempo de seguro, Idade do carro, Área do segurado, Idade do segurado, Modelo.
colsSelecionadasRF = ['Comprimento', 'Tempo de seguro', 'Idade do carro', 'Idade do segurado','Área do segurado', 'Modelo']

# Criando dataframe temporário apenas com as colunas selecionadas
dfRF = df

# Selecionando as colunas Categóricas
categorical_cols = dfRF.select_dtypes(include=['object']).columns

# Convertendo as colunas categóricas em variáveis de indicação
dfRF = pd.get_dummies(dfRF, columns=categorical_cols)

# Mostrar as variáveis de indicação criadas
dfRF.info(verbose = True)

# Atribuindo as colunas selecionadas a X
# x = dfRF
x = dfRF.drop(['is_claim'], axis = 1)

# Atribuindo a coluna alvo a Y
y = df["is_claim"]

# Fazendo a reamostragem para equilibrar os valores de Y
smt = SMOTEENN()
X_res, y_res = smt.fit_resample(x, y)
X_res = X_res[colsSelecionadasRF]

y_res.value_counts()

# Dividindo o dataset para treino e para teste, utilizando 20% do dataset para teste
x_train, x_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.20)

# Inicializando o algoritmo Random Forest
classifier = RandomForestClassifier()
# classifier = XGBClassifier()
 
#  Construindo uma 'floreste de árvores' da parte de treino
classifier.fit(x_train,y_train)

# Realizando as previsões
preds = classifier.predict(x_test)

# Criando o relatório com as principais métricas da classificação
report = classification_report(y_test, preds, output_dict=True)
df_report = pd.DataFrame(report).transpose()

st.table(df_report.style.format({'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}', 'support': '{:.2f}'}))

# Feature Importance
importance = classifier.feature_importances_
feature_importance = pd.DataFrame({'Feature': x_train.columns, 'Importance': importance})
feature_importance = feature_importance.sort_values('Importance', ascending=True).reset_index(drop=True)
feature_importance = feature_importance.tail(15)
print(feature_importance)
fig = px.bar(feature_importance, x='Importance', y='Feature', title='Feature Importance')

st.write(fig)

A1, A2 = st.columns(2)
cm = confusion_matrix(y_test, preds)
print(cm)

with A1:
   st.markdown("<h5 style='text-align: center;margin-bottom: 0px;'>Matriz de confusão</h5>", unsafe_allow_html=True)
   categories1 = ['Neg.', 'Pos.']
   categories2 = ['Neg.', 'Pos.']

   heatmap_fig = px.imshow(cm, x=categories1, y=categories2, color_continuous_scale='rdylgn')
   heatmap_fig.update_layout(
        xaxis={'title': 'Valores Preditos'},
        yaxis={'title': 'Valores Reais'},
       width=500, height=500
   )

   st.write(heatmap_fig)

with A2:
   st.markdown("<h5 style='text-align: center;margin-bottom: 0px;'>Pizza da matriz de confusão</h5>", unsafe_allow_html=True)
   cm_flat = cm.flatten()
   cm_fig = px.pie(values=cm_flat, names=['Verdadeiro Neg.','Falso Pos.','Falso Neg.','Verdadeiro Pos.'])
   cm_fig.update_traces(textinfo='percent+label')
   cm_fig.update_layout(showlegend=False)

   st.write(cm_fig)

# -----------------------------------------------------Regressão linear, KNN, Naive Bayes----------------------------------------------------- #
# # Selecionando algumas colunas
# # colunas_mantidas = ['is_claim', 'Idade do segurado', 'Idade do carro', 'Modelo', 'Tipo do combustível', 'Segmento', 'Tem assitência de freio', 'Tem câmera de ré', 'Tipo de transmissão']
# colunas_mantidas = ['Idade do segurado', 'Idade do carro', 'Modelo', 'Tipo do combustível', 'Segmento', 'Tem assitência de freio', 'Tem câmera de ré', 'Tipo de transmissão'] # Criei essa linha pq o pré-processamento não usa a variável alvo
# # dfKNN = df[colunas_mantidas]
# dfKNN = df #Essa linha é pra usar o dataset completo pra fazer o resample

# # Fazendo o tratamento das colunas categóricas para fazer o resample
# # Selecionando as colunas Categóricas
# categorical_cols = dfKNN.select_dtypes(include=['object']).columns

# # Convertendo as colunas categóricas em variáveis de indicação
# dfKNN = pd.get_dummies(dfKNN, columns=categorical_cols)

# # Mostrar as variáveis de indicação criadas
# # dfKNN.info(verbose = True)

# #Setando is_claim como y e x com o resto das colunas selecionadas
# y = df['is_claim']
# x = dfKNN.drop('is_claim', axis = 1) # Só troquei pra o dfKNN

# # Fazendo o Resample
# smt = SMOTEENN()
# x_res, y_res = smt.fit_resample(x, y)
# x_res = x_res[colunas_mantidas] # Pegando apenas as colunas desejadas após o resample
# y_res.value_counts()


# x_treino, x_teste, y_treino, y_teste = train_test_split(x_res, y_res, test_size = 0.2, random_state = 1)

# #Treinando o ml para regressão linear, KNN e Naive Bayes
# modeloRegressaoLinear = LinearRegression()
# modeloKNN = KNeighborsClassifier(n_neighbors=2)
# modeloNaiveBayes = MultinomialNB()

# # modeloRegressaoLinear.fit(x_treino, y_treino)
# modeloKNN.fit(x_treino, y_treino)
# # modeloNaiveBayes.fit(x_treino, y_treino)

# # previsaoRegressaoLinear = modeloRegressaoLinear.predict(x_teste)
# previsaoKNN = modeloKNN.predict(x_teste)
# # previsaoNaiveBayes = modeloNaiveBayes.predict(x_teste)

# # k_range = range(1, 21)
# # scores = []

# # for k in k_range:
# #    modeloKNN = KNeighborsClassifier(n_neighbors=k)
# #    modeloKNN.fit(x_treino, y_treino)
# #    previsaoKNN = modeloKNN.predict(x_teste)
# #    scores.append(metrics.accuracy_score(y_teste, previsaoKNN))

# # for i, item in enumerate(scores):
# #    st.write(f"{i+1} vizinhos: {item}")

# # #Mostra o resultado da avaliação
# # #P/ Regressão linear
# # st.title("Resultado da avaliação para regressão linear")
# # st.write("accuracy =", metrics.r2_score(y_teste, previsaoRegressaoLinear))

# #P/ KNN
# st.title("Resultado da avaliação para KNN")
# report = classification_report(y_teste, previsaoKNN, output_dict=True)
# df_report = pd.DataFrame(report).transpose()
# st.table(df_report.style.format({'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}', 'support': '{:.2f}'}))

# # #P/ Naive Bayes
# # st.title("Resultado da avaliação para Naive Bayes")
# # report = classification_report(y_teste, previsaoNaiveBayes, output_dict=True)
# # df_report = pd.DataFrame(report).transpose()
# # st.table(df_report.style.format({'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}', 'support': '{:.2f}'}))

# # st.title("Matriz de confusão")
# # # Definindo o limite de decisão
# # limite = 0.5

# # # Obtendo as previsões binárias com base no limite de decisão
# # previsao_binaria = np.where(previsaoRegressaoLinear >= limite, 1, 0)

# # # Criando a matriz de confusão para regressão linear
# # matriz_confusao = metrics.confusion_matrix(y_teste, previsao_binaria)

# # # Mostrando a matriz de confusão para regressão linear
# # A1, A2 = st.columns(2)

# # with A1:
# #     st.markdown("<h5 style='text-align: center;margin-bottom: 0px;'>Matriz de confusão para regressão linear</h5>", unsafe_allow_html=True)
# #     categories1 = ['Neg.', 'Pos.']
# #     categories2 = ['Neg.', 'Pos.']

# #     heatmap_fig = px.imshow(matriz_confusao, x=categories1, y=categories2, color_continuous_scale='rdylgn')
# #     heatmap_fig.update_layout(
# #         xaxis={'title': 'Valores Preditos'},
# #         yaxis={'title': 'Valores Reais'},
# #         width=500, height=500
# #     )

# #     st.write(heatmap_fig)

# # with A2:
# #    st.markdown("<h5 style='text-align: center;margin-bottom: 0px;'>Pizza da matriz de confusão para regressão linear</h5>", unsafe_allow_html=True)
# #    cm_flat = matriz_confusao.flatten()
# #    cm_fig = px.pie(values=cm_flat, names=['Verdadeiro Neg.','Falso Pos.','Falso Neg.','Verdadeiro Pos.'])
# #    cm_fig.update_traces(textinfo='percent+label')
# #    cm_fig.update_layout(showlegend=False)

# #    st.write(cm_fig)

# # Criando a matriz de confusão para KNN
# matriz_confusao = metrics.confusion_matrix(y_teste, previsaoKNN)

# # Mostrando a matriz de confusão para KNN
# A1, A2 = st.columns(2)

# with A1:
#     st.markdown("<h5 style='text-align: center;margin-bottom: 0px;'>Matriz de confusão para KNN</h5>", unsafe_allow_html=True)
#     categories1 = ['Neg.', 'Pos.']
#     categories2 = ['Neg.', 'Pos.']

#     heatmap_fig = px.imshow(matriz_confusao, x=categories1, y=categories2, color_continuous_scale='rdylgn')
#     heatmap_fig.update_layout(
#         xaxis={'title': 'Valores Preditos'},
#         yaxis={'title': 'Valores Reais'},
#         width=500, height=500
#     )

#     st.write(heatmap_fig)

# with A2:
#    st.markdown("<h5 style='text-align: center;margin-bottom: 0px;'>Pizza da matriz de confusão para KNN</h5>", unsafe_allow_html=True)
#    cm_flat = matriz_confusao.flatten()
#    cm_fig = px.pie(values=cm_flat, names=['Verdadeiro Neg.','Falso Pos.','Falso Neg.','Verdadeiro Pos.'])
#    cm_fig.update_traces(textinfo='percent+label')
#    cm_fig.update_layout(showlegend=False)

#    st.write(cm_fig)

# # # Criando a matriz de confusão para Naive Bayer
# # matriz_confusao = metrics.confusion_matrix(y_teste, previsaoNaiveBayes)

# # # Mostrando a matriz de confusão para Naive Bayer
# # A1, A2 = st.columns(2)

# # with A1:
# #     st.markdown("<h5 style='text-align: center;margin-bottom: 0px;'>Matriz de confusão para Naive Bayes</h5>", unsafe_allow_html=True)
# #     categories1 = ['Neg.', 'Pos.']
# #     categories2 = ['Neg.', 'Pos.']

# #     heatmap_fig = px.imshow(matriz_confusao, x=categories1, y=categories2, color_continuous_scale='rdylgn')
# #     heatmap_fig.update_layout(
# #         xaxis={'title': 'Valores Preditos'},
# #         yaxis={'title': 'Valores Reais'},
# #         width=500, height=500
# #     )

# #     st.write(heatmap_fig)

# # with A2:
# #    st.markdown("<h5 style='text-align: center;margin-bottom: 0px;'>Pizza da matriz de confusão para Naive Bayes</h5>", unsafe_allow_html=True)
# #    cm_flat = matriz_confusao.flatten()
# #    cm_fig = px.pie(values=cm_flat, names=['Verdadeiro Neg.','Falso Pos.','Falso Neg.','Verdadeiro Pos.'])
# #    cm_fig.update_traces(textinfo='percent+label')
# #    cm_fig.update_layout(showlegend=False)

# #    st.write(cm_fig)

# PARA CENTRALIZAR OS GRÁFICOS E TABELAS NA PÁGINA (MANTER SEMPRE NO FINAL DO ARQUIVO)
st.markdown("""
   <style>
   .element-container {
       display: flex;
       justify-content: center;
       align-items: center;
   }
   </style>
   """, unsafe_allow_html=True)