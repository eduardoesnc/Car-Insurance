import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
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
from imblearn.under_sampling import TomekLinks
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC


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
#Definindo a semente aleatória para garantir que os resultados sejam reprodutíveis
seed = 42
#Definindo a taxa de teste
test_ratio = 0.2

# Leitura e tratamento dos dados
@st.cache_data
def readData():
    dataset = pd.read_csv('./data/train.csv')
    datasetPlus = pd.read_csv('./data/test.csv')
    sampleSub = pd.read_csv('./data/sample_submission.csv')
    return dataset, datasetPlus, sampleSub

df, dfPlus, sample = readData()

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

# Chamando o modelo treinado
with open("model.pkl", "rb") as f:
        classifier  = pickle.load(f)

df = tratarDados(df)
dfPlusRF = dfPlus.copy()
sample = sample.drop(['policy_id'], axis=1)
dfPlusRF = pd.concat([pd.DataFrame(dfPlusRF), pd.DataFrame(sample)], axis=1)
dfPlusRF = tratarDados(dfPlusRF)
df = pd.concat([pd.DataFrame(df), pd.DataFrame(dfPlusRF)], axis = 0)

st.subheader('Seleção das Features')
st.markdown("""Realizamos teste e seleção das features do sistema. Avaliamos todas as características em relação à sua contribuição
               para a eficácia do sistema. Selecionamos as features mais importantes, otimizando o desempenho do modelo e garantindo
               sua eficiência e escalabilidade. Com isso, as features selecionadas foram Comprimento, Tempo de seguro, Idade do carro,
               Área do segurado, Idade do segurado, Modelo.""")

# Selecionando as colunas Categóricas
categorical_cols = df.select_dtypes(include=['object']).columns

# Convertendo as colunas categóricas em variáveis de indicação
df = pd.get_dummies(df, columns=categorical_cols)

# # Mostrar as variáveis de indicação criadas
df.info(verbose = True)

# Retirando a coluna alvo
x = df.drop(['is_claim'], axis = 1)

# Atribuindo a coluna alvo a Y
y = df["is_claim"]

# Fazendo a reamostragem para equilibrar os valores de Y
smt = SMOTEENN()
X_res, y_res = smt.fit_resample(x, y)

# Selecionando as colunas Tempo de seguro, Idade do carro, Idade do segurado, Área do segurado, Comprimento, Modelo.
colsSelecionadas = ['Tempo de seguro', 'Idade do carro', 'Idade do segurado', 'Área do segurado', 'Comprimento', 'Modelo']
X_res = X_res[colsSelecionadas]

data = pd.concat([pd.DataFrame(X_res), pd.DataFrame(y_res)], axis=1)
dataTrue = data[data['is_claim'] == 1]
dataFalse = data[data['is_claim'] == 0]

option = st.selectbox(
    'Seleciona um para comparar o comportamento deles em relação a classificação Verdadeira ou Falsa:',
    colsSelecionadas)

option = colsSelecionadas.index(option)

A1, A2 = st.columns(2)

with A1:
    st.markdown("<h5 style='text-align: center;margin-bottom: -34px;'>Verdadeira</h5>", unsafe_allow_html=True)
    fig = px.histogram(dataTrue, x=colsSelecionadas[option], width= 550)
    st.write(fig)

with A2: 
    st.markdown("<h5 style='text-align: center;margin-bottom: -34px;'>Falsa</h5>", unsafe_allow_html=True)
    fig = px.histogram(dataFalse, x=colsSelecionadas[option], width= 550)
    st.write(fig)

st.markdown("---")
# -----------------------------------------------------RANDOM FOREST----------------------------------------------------- #
st.subheader("Random Forest")
st.markdown("""<p style="font-size: 16px;">
           Features selecionadas: Tempo de seguro, Idade do carro, Idade do segurado, Área do segurado, Comprimento, Modelo.
           </p>""", unsafe_allow_html=True)

if st.checkbox('Mostrar código'):
   st.code("""
# Selecionando as colunas Tempo de seguro, Idade do carro, Idade do segurado, Área do segurado, Comprimento, Modelo.
colsSelecionadas = ['Tempo de seguro', 'Idade do carro', 'Idade do segurado', 'Área do segurado', 'Comprimento', 'Modelo']

# Selecionando as colunas Categóricas
categorical_cols = dfRF.select_dtypes(include=['object']).columns

# Convertendo as colunas categóricas em variáveis de indicação
dfRF = pd.get_dummies(dfRF, columns=categorical_cols)

# Retirando a coluna alvo
x = dfRF.drop(['is_claim'], axis = 1)

# Atribuindo a coluna alvo a Y
y = dfRF["is_claim"]

# Fazendo a reamostragem para equilibrar os valores de Y
smt = SMOTEENN()
X_res, y_res = smt.fit_resample(x, y)

#Separando apenas as colunas que iremos utilizar
X_res = X_res[colsSelecionadas]

# Dividindo o dataset para treino e para teste, utilizando 20% do dataset para teste
x_train, x_test, y_train, y_test = train_test_split(X_res, y_res, test_size=test_ratio, random_state=seed)

# Inicializando o algoritmo Random Forest
classifier = RandomForestClassifier()

#  Construindo uma 'floreste de árvores' da parte de treino
classifier.fit(x_train,y_train)
""")

# Dividindo o dataset para treino e para teste, utilizando 20% do dataset para teste
x_train, x_test, y_train, y_test = train_test_split(X_res, y_res, test_size=test_ratio, random_state=seed)

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

# #Treinando o ml para regressão linear, KNN e Naive Bayes
# modeloRegressaoLogistica = LogisticRegression()
# modeloKNN = KNeighborsClassifier(n_neighbors=6)
# modeloNaiveBayes = MultinomialNB()
# modeloGradientBoost = GradientBoostingClassifier()

# modeloRegressaoLogistica.fit(x_train, y_train)
# modeloKNN.fit(x_train, y_train)
# modeloNaiveBayes.fit(x_train, y_train)
# modeloGradientBoost.fit(x_train, y_train)

# previsaoRegressaoLogistica = modeloRegressaoLogistica.predict(x_test)
# previsaoKNN = modeloKNN.predict(x_test)
# previsaoNaiveBayes = modeloNaiveBayes.predict(x_test)
# previsaoGB = modeloGradientBoost.predict(x_test)

# # k_range = range(1, 21)
# # scores = []

# # for k in k_range:
# #    modeloKNN = KNeighborsClassifier(n_neighbors=k)
# #    modeloKNN.fit(x_train, y_train)
# #    previsaoKNN = modeloKNN.predict(x_test)
# #    scores.append(metrics.accuracy_score(y_test, previsaoKNN))

# # for i, item in enumerate(scores):
# #    st.write(f"{i+1} vizinhos: {item}")

# #Mostra o resultado da avaliação
# #P/ Regressão logística
# st.title("Resultado da avaliação para regressão logística")
# report = classification_report(y_test, previsaoRegressaoLogistica, output_dict=True)
# df_report = pd.DataFrame(report).transpose()
# st.table(df_report.style.format({'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}', 'support': '{:.2f}'}))

# #P/ KNN
# st.title("Resultado da avaliação para KNN")
# report = classification_report(y_test, previsaoKNN, output_dict=True)
# df_report = pd.DataFrame(report).transpose()
# st.table(df_report.style.format({'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}', 'support': '{:.2f}'}))

# # Criando a matriz de confusão para KNN
# matriz_confusao = metrics.confusion_matrix(y_test, previsaoKNN)

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

# #P/ Naive Bayes
# st.title("Resultado da avaliação para Naive Bayes")
# report = classification_report(y_test, previsaoNaiveBayes, output_dict=True)
# df_report = pd.DataFrame(report).transpose()
# st.table(df_report.style.format({'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}', 'support': '{:.2f}'}))

# st.title("Matriz de confusão")

# # Criando a matriz de confusão para regressão logística
# matriz_confusao = metrics.confusion_matrix(y_test, previsaoRegressaoLogistica)

# # Mostrando a matriz de confusão para regressão logística
# A1, A2 = st.columns(2)

# with A1:
#     st.markdown("<h5 style='text-align: center;margin-bottom: 0px;'>Matriz de confusão para regressão logística</h5>", unsafe_allow_html=True)
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
#    st.markdown("<h5 style='text-align: center;margin-bottom: 0px;'>Pizza da matriz de confusão para regressão logística</h5>", unsafe_allow_html=True)
#    cm_flat = matriz_confusao.flatten()
#    cm_fig = px.pie(values=cm_flat, names=['Verdadeiro Neg.','Falso Pos.','Falso Neg.','Verdadeiro Pos.'])
#    cm_fig.update_traces(textinfo='percent+label')
#    cm_fig.update_layout(showlegend=False)

#    st.write(cm_fig)

# # Criando a matriz de confusão para Naive Bayer
# matriz_confusao = metrics.confusion_matrix(y_test, previsaoNaiveBayes)

# # Mostrando a matriz de confusão para Naive Bayer
# A1, A2 = st.columns(2)

# with A1:
#     st.markdown("<h5 style='text-align: center;margin-bottom: 0px;'>Matriz de confusão para Naive Bayes</h5>", unsafe_allow_html=True)
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
#    st.markdown("<h5 style='text-align: center;margin-bottom: 0px;'>Pizza da matriz de confusão para Naive Bayes</h5>", unsafe_allow_html=True)
#    cm_flat = matriz_confusao.flatten()
#    cm_fig = px.pie(values=cm_flat, names=['Verdadeiro Neg.','Falso Pos.','Falso Neg.','Verdadeiro Pos.'])
#    cm_fig.update_traces(textinfo='percent+label')
#    cm_fig.update_layout(showlegend=False)

#    st.write(cm_fig)


# st.title("Resultado da avaliação para Gradient Boosting")
# report = classification_report(y_test, previsaoGB, output_dict=True)
# df_report = pd.DataFrame(report).transpose()
# st.table(df_report.style.format({'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}', 'support': '{:.2f}'}))


# # Criando a matriz de confusão para KNN
# matriz_confusao = metrics.confusion_matrix(y_test, previsaoGB)

# # Mostrando a matriz de confusão para KNN
# A1, A2 = st.columns(2)

# with A1:
#     st.markdown("<h5 style='text-align: center;margin-bottom: 0px;'>Matriz de confusão para Gradient Boosting</h5>", unsafe_allow_html=True)
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
#    st.markdown("<h5 style='text-align: center;margin-bottom: 0px;'>Pizza da matriz de confusão para Gradient Boosting</h5>", unsafe_allow_html=True)
#    cm_flat = matriz_confusao.flatten()
#    cm_fig = px.pie(values=cm_flat, names=['Verdadeiro Neg.','Falso Pos.','Falso Neg.','Verdadeiro Pos.'])
#    cm_fig.update_traces(textinfo='percent+label')
#    cm_fig.update_layout(showlegend=False)

#    st.write(cm_fig)

# -----------------------------------------------------SVM - Support Vector Machine ----------------------------------------------------- #

# Dividindo o dataset para treino e para teste, utilizando 20% do dataset para teste
#x_train, x_test, y_train, y_test = train_test_split(X_res, y_res, test_size=test_ratio, random_state=seed)

#Instanciando o classificador SVM:
#classifierSvm = SVC(kernel = 'rbf', gamma = 4)
#classifierSvm.fit(x_train,y_train)

# Realizando as previsões com o SVM
#prev_svm = classifierSvm.predict(x_test)

# Criando o relatório com as principais métricas da classificação
#report = classification_report(y_test, prev_svm, output_dict=True)
#df_report = pd.DataFrame(report).transpose()

#st.table(df_report.style.format({'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}', 'support': '{:.2f}'}))

#A1, A2 = st.columns(2)
#cm = confusion_matrix(y_test, prev_svm)
#print(cm)

#with A1:
#   st.markdown("<h5 style='text-align: center;margin-bottom: 0px;'>Matriz de confusão</h5>", unsafe_allow_html=True)
#   categories1 = ['Neg.', 'Pos.']
#   categories2 = ['Neg.', 'Pos.']

#   heatmap_fig = px.imshow(cm, x=categories1, y=categories2, color_continuous_scale='rdylgn')
#   heatmap_fig.update_layout(
#        xaxis={'title': 'Valores Preditos'},
#        yaxis={'title': 'Valores Reais'},
#       width=500, height=500
#   )

#   st.write(heatmap_fig)

#with A2:
#   st.markdown("<h5 style='text-align: center;margin-bottom: 0px;'>Pizza da matriz de confusão</h5>", unsafe_allow_html=True)
#   cm_flat = cm.flatten()
#   cm_fig = px.pie(values=cm_flat, names=['Verdadeiro Neg.','Falso Pos.','Falso Neg.','Verdadeiro Pos.'])
#   cm_fig.update_traces(textinfo='percent+label')
#   cm_fig.update_layout(showlegend=False)

#    st.write(cm_fig)


#------------------------------------------------------XGBoost Classifier------------------------------------------------------#

# Dividindo o dataset para treino e para teste, utilizando 20% do dataset para teste
# Manter essa linha comentada se for fazer comparação com outro modelo, assim eles usaram a mesma base de treino e teste
# x_train, x_test, y_train, y_test = train_test_split(X_res, y_res, test_size=test_ratio, random_state=seed)

# Instanciando o classificador XGBoost e ajustando o modelo aos dados de treinamento
xgb_classifier = XGBClassifier(learning_rate = 1)
xgb_classifier.fit(x_train, y_train)

# Realizando as previsões com o XGBoost
prev_xgb = xgb_classifier.predict(x_test)

# Criando o relatório com as principais métricas da classificação
report = classification_report(y_test, prev_xgb, output_dict=True)
df_report = pd.DataFrame(report).transpose()

st.table(df_report.style.format({'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}', 'support': '{:.2f}'}))

# Feature Importance
importance = xgb_classifier.feature_importances_
feature_importance = pd.DataFrame({'Feature': x_train.columns, 'Importance': importance})
feature_importance = feature_importance.sort_values('Importance', ascending=True).reset_index(drop=True)
feature_importance = feature_importance.tail(15)
print(feature_importance)
fig = px.bar(feature_importance, x='Importance', y='Feature', title='Feature Importance')

st.write(fig)

A1, A2 = st.columns(2)
cm = confusion_matrix(y_test, prev_xgb)
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

#-------------------------------------------------------------------------------------------------------------------------------------#

st.markdown("---")
# Botão para página de Estimar Chance
st.markdown("""
    <br><br>
    <div style="text-align: center; margin-top: 60px;">
    <a href="/Estimar_Chance" target="_self"
    style="text-decoration: none;
            color: white;
            font-size: 18px;
            font-weight: 550;
            background: rgb(243,68,55);
            background: linear-gradient(156deg, rgba(243,68,55,1) 30%, rgba(249,170,61,1) 70%);
            padding: 15px 40px;
            border-radius: 8px;">
    Estimar Chance de Reivindicação
    </a>
    </div>
    """, unsafe_allow_html=True)

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