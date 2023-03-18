import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.feature_selection import mutual_info_classif
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTENC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


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
    database = database.drop(['length', 'width', 'height'], axis=1)

    # Normalizar policy tenure com min max normalization
    # policy_df = database['policy_tenure']
    # database['policy_tenure'] = (policy_df - policy_df.min()) / (policy_df.max() - policy_df.min())

    age_of_car_outliers = database.age_of_car > database.age_of_car.quantile(0.995)
    database = database.loc[~age_of_car_outliers]

    age_of_policyholder_outliers = database.age_of_policyholder > database.age_of_policyholder.quantile(0.995)
    database = database.loc[~age_of_policyholder_outliers]

    database = database.replace({ "No" : False , "Yes" : True })
    
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


df.rename(columns={'policy_tenure': 'Tempo de seguro', 'turning_radius': 'Espaço necessário para curva',
                   'age_of_car': 'Idade do carro', 'volume': 'Volume', 'population_density': 'Densidade populacional',
                   'area_cluster': 'Área do segurado', 'age_of_policyholder': 'Idade do segurado',
                   'engine_type': 'Tipo do motor', 'model': 'Modelo', 'gross_weight': 'Peso máximo',
                   'displacement': 'cilindradas (cc)', 'max_torque': 'Torque máximo', 'max_power': 'Força máxima',
                   'segment': 'Segmento', 'is_adjustable_steering': 'Volante ajustável?',
                   'cylinder': 'Quantidade de cilindros', 'is_front_fog_lights': 'Tem luz de neblina?',
                   'is_brake_assist': 'Tem assitência de freio',
                   'is_driver_seat_height_adjustable': 'Banco do motorista é ajustável?',
                   'fuel_type': 'Tipo do combustível'}, inplace=True)

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
            Features selecionadas: Volume, Tempo de seguro, Idade do carro, Área do segurado, Densidade populacional, Idade do segurado, Modelo.
            </p>""", unsafe_allow_html=True)

if st.checkbox('Mostrar código'):
    st.code("""
# Selecionando as colunas Volume, Tempo de seguro, Idade do carro, Área do segurado, Densidade populacional, Idade do segurado, Modelo.
colsSelecionadasRF = ['Volume', 'Tempo de seguro', 'Idade do carro', 'Área do segurado', 'Densidade populacional',
                      'Idade do segurado', 'Modelo']

#Criando dataframe temporário apenas com as colunas selecionadas
tempDF = df[colsSelecionadasRF]

#Selecionando as colunas Categóricas
categorical_cols = tempDF.select_dtypes(include=['object']).columns

# Convertendo as colunas categóricas em variáveis de indicação
dfRF = pd.get_dummies(tempDF, columns=categorical_cols)

# Atribuindo as colunas selecionadas a X
x = dfRF

# Atribuindo a coluna alvo a Y
y = df["is_claim"]

# Fazendo a reamostragem para equilibrar os valores de Y
smt = SMOTEENN()
X_res, y_res = smt.fit_resample(x, y)
y_res.value_counts()

# Dividindo o dataset para treino e para teste, utilizando 20% do dataset para teste
x_train, x_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.20)

# Inicializando o algoritmo Random Forest
classifier = RandomForestClassifier()

# Construindo uma 'floreste de árvores' da parte de treino
classifier.fit(x_train,y_train)

# Realizando as previsões
preds = classifier.predict(x_test)

# Criando o relatório com as principais métricas da classificação
report = classification_report(y_test, preds, output_dict=True)
df_report = pd.DataFrame(report).transpose()
st.table(df_report.style.format({'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}', 'support': '{:.2f}'}))
""")

# Selecionando as colunas Volume, Tempo de seguro, Idade do carro, Área do segurado, Densidade populacional, Idade do segurado, Modelo.
colsSelecionadasRF = ['Volume', 'Tempo de seguro', 'Idade do carro', 'Área do segurado', 'Densidade populacional', 'Idade do segurado', 'Modelo']

#Criando dataframe temporário apenas com as colunas selecionadas
tempDF = df[colsSelecionadasRF]

#Selecionando as colunas Categóricas
categorical_cols = tempDF.select_dtypes(include=['object']).columns

# Convertendo as colunas categóricas em variáveis de indicação
dfRF = pd.get_dummies(tempDF, columns=categorical_cols)

# Mostrar as variáveis de indicação criadas
# dfRF.info(verbose = True)

# Atribuindo as colunas selecionadas a X
x = dfRF

# Atribuindo a coluna alvo a Y
y = df["is_claim"]

# Fazendo a reamostragem para equilibrar os valores de Y
smt = SMOTEENN()
X_res, y_res = smt.fit_resample(x, y)
y_res.value_counts()

# Dividindo o dataset para treino e para teste, utilizando 20% do dataset para teste
x_train, x_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.20)

# Inicializando o algoritmo Random Forest
classifier = RandomForestClassifier()
 
 # Construindo uma 'floreste de árvores' da parte de treino
classifier.fit(x_train,y_train)

# Realizando as previsões
preds = classifier.predict(x_test)

# Criando o relatório com as principais métricas da classificação
report = classification_report(y_test, preds, output_dict=True)
df_report = pd.DataFrame(report).transpose()

st.table(df_report.style.format({'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}', 'support': '{:.2f}'}))

# Feature Importance
importance = classifier.feature_importances_
feature_importance = pd.DataFrame({'Feature': x.columns, 'Importance': importance})
feature_importance = feature_importance.sort_values('Importance', ascending=True).reset_index(drop=True)
feature_importance = feature_importance.tail(15)
# print(feature_importance)
fig = px.bar(feature_importance, x='Importance', y='Feature', title='Feature Importance')

st.write(fig)

A1, A2 = st.columns(2)
cm = confusion_matrix(y_test, preds)
# print(cm)

with A1:
    st.markdown("<h5 style='text-align: center;margin-bottom: 0px;'>Matriz de confusão</h5>", unsafe_allow_html=True)
    categories1 = ['Neg.', 'Pos.']
    categories2 = ['Neg.', 'Pos.']

    heatmap_fig = px.imshow(cm, x=categories1, y=categories2, color_continuous_scale='rdylgn')
    heatmap_fig.update_layout(
        xaxis={'title': 'Valores Reais'},
        yaxis={'title': 'Valores Preditos'},
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

st.markdown("""
    <style>
    .element-container {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)
