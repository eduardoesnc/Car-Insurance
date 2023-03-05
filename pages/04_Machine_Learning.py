import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import plotly.express as px
from sklearn.feature_selection import mutual_info_classif

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
    initial_sidebar_state='expanded'
)


# Leitura e tratamento dos dados
@st.cache_data
def readData():
    dataset = pd.read_csv('./data/train.csv')
    return dataset


df = readData()


def tratarDados(database):
    # Os seguintes dados já estão normalizados pela formula de min-max, tem que achar um jeito de reverter se necessário
    # database['age_of_policyholder'] = round(database['age_of_policyholder'].mul(100))
    # database['age_of_car'] = round(database['age_of_car'].mul(100))

    # Apagar coluna policy_id, já que são apenas IDs
    database = database.drop(['policy_id'], axis=1)

    # Tranformar as colunas largura, tamanho e altura em apenas uma coluna chamada volume
    database['volume'] = np.log(database.length * database.width * database.height * 1e-6)
    database = database.drop(['length', 'width', 'height'], axis=1)

    # Normalizar policy tenure com min max normalization
    policy_df = database['policy_tenure']
    normPolicy = (policy_df - policy_df.min()) / (policy_df.max() - policy_df.min())
    pd.concat([normPolicy, database['is_claim']], axis=1)

    return database


st.title('Desenvolvimento do Macinhe Learning')

df = tratarDados(df)

# Gráfico Mutual Information Score
st.subheader("Mutual Information Score")
st.markdown("""<p style="font-size: 16px;text-align: center; margin-top: 0px">
            O Mutual Information Score é uma métrica de aprendizado de máquina que mede a dependência entre duas
            variáveis aleatórias. No caso do dataset que estamos trabalhando, estamos usando essa métrica para 
            avaliar a relação entre as variáveis do dataset e a coluna is_claim e buscar quais seriam as mais úteis para
            desenvolver o Machine Learning.
            </p>""", unsafe_allow_html=True)


def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category", "bool"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    all_mi_scores = []
    for random_state in range(0, 5):
        miScores = mutual_info_classif(X, y, discrete_features=discrete_features, random_state=random_state)
        all_mi_scores.append(miScores)
    all_mi_scores = np.mean(all_mi_scores, axis=0)
    all_mi_scores = pd.Series(all_mi_scores, name="MI Scores", index=X.columns)
    all_mi_scores = all_mi_scores.sort_values(ascending=False)
    return all_mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    ticks = list(scores.index)
    fig = px.bar(y=ticks, x=scores, labels={'x': '', 'y': ''}, height=600, width=1000)
    st.write(fig)


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

y_target = df.is_claim.astype('int16')

mi_scores = make_mi_scores(df.drop('is_claim', axis=1), y_target)

# print(mi_scores.head(20))
# print(mi_scores.tail(20))
plt.figure(dpi=100, figsize=(8, 5))
st.set_option('deprecation.showPyplotGlobalUse', False)
plot_mi_scores(mi_scores.head(20))

st.markdown("---")
st.subheader("XGBoost Classifier")

st.markdown("""
    <style>
    .element-container {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)
