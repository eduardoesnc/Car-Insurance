import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import plotly.express as px
from sklearn.feature_selection import mutual_info_classif

st.set_page_config(
    page_title="EDA",
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


# st.sidebar.header('Dashboard')


# Leitura e tratamento dos dados
# @st.cache_data
def readData():
    dataset = pd.read_csv('./data/train.csv')
    return dataset


bf = readData()


def tratarDados(database):
    # Os seguintes dados já estão normalizados pela formula de min-max, tem que achar um jeito de reverter se necessário
    # database['age_of_policyholder'] = round(database['age_of_policyholder'].mul(100))
    # database['age_of_car'] = round(database['age_of_car'].mul(100))

    # Separando as tabelas max_torque e max_power
    # database["max_torque_Nm"] = database['max_torque'] \
    #     .str.extract(r"([-+]?[0-9]*\.?[0-9]+)(?=\s*Nm)").astype('float64')
    # database["max_torque_rpm"] = database['max_torque'] \
    #     .str.extract(r"([-+]?[0-9]*\.?[0-9]+)(?=\s*rpm)").astype('float64')
    #
    # database["max_power_bhp"] = database['max_power'].str.extract(r"([-+]?[0-9]*\.?[0-9]+)(?=\s*bhp)").astype('float64')
    # database["max_power_rpm"] = database['max_power'].str.extract(r"([-+]?[0-9]*\.?[0-9]+)(?=\s*rpm)").astype('float64')
    #
    # database = database.drop(['max_torque'], axis=1)
    # database = database.drop(['max_power'], axis=1)

    # Apagar coluna policy_id, já que são apenas IDs
    database = database.drop(['policy_id'], axis=1)

    # Tranformar as colunas largura tamanho e altura em apenas uma coluna chamada volume
    database['volume'] = np.log(database.length * database.width * database.height * 1e-6)
    database = database.drop(['length', 'width', 'height'], axis=1)

    # Normalizar policy tenure com min max normalization
    policy_df = bf['policy_tenure']
    normPolicy = (policy_df - policy_df.min()) / (policy_df.max() - policy_df.min())
    pd.concat([normPolicy, bf['is_claim']], axis=1)

    return database


# Criação de array com o nome de todas as colunas para facilitar na criação dos filtros
dict_nome_colunas = ['Idade do carro em anos', 'Idade do segurado em anos', 'Área do segurado',
                     'Densidade populacional',
                     'Código da fabricante do carro', 'Segmento do carro (A / B1 / B2 / C1 / C2)', 'Modelo do carro',
                     'Tipo de combustível usado no carro', 'Torque máximo gerado pelo carro (Nm@rpm)',
                     'Força máxima gerada pelo carro (bhp@rpm)',
                     'Tipo de motor usado pelo carro', 'Número de airbags instalados no carro',
                     'Tem controle de estabilização eletrônica?',
                     'O volante é ajustável?', 'Tem sistema de monitoramento da pressão do pneu?',
                     'Tem sensores de ré?',
                     'Tem câmera de ré?', 'Tipo de freio usado no carro', 'Cilindradas do motor (cc)',
                     'Quantidade de cilindros do carro',
                     'Tipo de transmissão do carro', 'Quantidade de marchas do carro', 'Tipo de direção do carro',
                     'Espaço necessário pro carro fazer uma certa curva',
                     'Volume do carro', 'Peso máximo suportado pelo carro',
                     'Tem farol de neblina?', 'Tem limpador de vidro traseiro?', 'Tem desembaçador de vidro traseiro?',
                     'Tem assistência de freio?',
                     'Tem trava elétrica de porta?', 'Tem direção hidráulica?', 'O acento do motorista é ajustável?',
                     'Tem espelho de retrovisor traseiro?',
                     'Tem luz indicativa de problemas no motor?', 'Tem sistema de alerta de velocidade?',
                     'Classificação de segurança pela NCAP (de 0 a 5)']
nome_colunas = ['age_of_car', 'age_of_policyholder', 'area_cluster', 'population_density', 'make', 'segment',
                'model', 'fuel_type', 'max_torque_Nm', 'max_power_bhp', 'engine_type', 'airbags', 'is_esc',
                'is_adjustable_steering', 'is_tpms',
                'is_parking_sensors', 'is_parking_camera', 'rear_brakes_type', 'displacement', 'cylinder',
                'transmission_type', 'gear_box', 'steering_type',
                'turning_radius', 'volume', 'gross_weight', 'is_front_fog_lights', 'is_rear_window_wiper',
                'is_rear_window_washer',
                'is_rear_window_defogger', 'is_brake_assist', 'is_power_door_locks', 'is_central_locking',
                'is_power_steering', 'is_driver_seat_height_adjustable', 'is_day_night_rear_view_mirror',
                'is_ecw', 'is_speed_alert', 'ncap_rating', 'is_claim']


# Verificando se há valores nulos
@st.cache_data
def manterDados(database):
    null = database.isnull().sum()
    return null


st.title('Ánalise de dados com o dataset Car-Insurance')


# Download do dataset
@st.cache_data
def convert_df(database):
    return database.to_csv().encode('utf-8')


csv = convert_df(bf)

st.download_button(
    label="Baixar dataset",
    data=csv,
    file_name='car-insurance.csv',
    mime='text/csv',
)

bf = tratarDados(bf)  # tbf é o dataset com tratamento de dados
numericos = bf.select_dtypes(include=[np.float64, np.int64])
categoricos = bf.select_dtypes(include=[np.object])

# ___________________________________________________#

st.markdown(
    """
    - [Fonte](https://www.kaggle.com/datasets/ifteshanajnin/carinsuranceclaimprediction-classification)
    - [Dicionário de dados](https://github.com/eduardoesnc/SMD/blob/Streamlit/data/Dicionário%20de%20dados.pdf)
    """, unsafe_allow_html=True
)

st.markdown("---")

if st.checkbox('Mostrar dataset'):
    st.subheader('Dataset')
    st.dataframe(bf)

# if st.checkbox('Mostrar média dos valores das colunas'):
#     st.subheader('Média dos valores das colunas')
#     st.table(manterDados(bf))

# SIDEBAR
# if st.sidebar.checkbox('Mostrar os tipos dos dados do dataset'):
#     st.subheader('Tipos dos dados')
#     var = bf.dtypes
#     st.write(var)

# st.sidebar.markdown("---")

# _____________GRÁFICOS USANDO STREAMLIT E PLOTLY________________ #
st.header('Análises')

option = st.selectbox(
    'Seleciona um para comparar com a possibilidade de reivindicação dentro de 6 meses:',
    dict_nome_colunas)

option = dict_nome_colunas.index(option)

fig = px.bar(bf, x=nome_colunas[option], y='is_claim',
             labels={nome_colunas[option]: dict_nome_colunas[option], 'is_claim': 'Reinvidicado'})
st.write(fig)

st.caption('Gráficos para analisar como os valores das colunas interagem com a chance de reivindicação')
st.markdown("---")

# selecoes = st.sidebar.multiselect('Escolha duas opções e o tipo de gráfico desejado', nome_colunas)
# tipoGrafico = st.sidebar.radio('', ['Linha', 'Barra'])
# if len(selecoes) == 2:
#     st.subheader("Gráficos formados pelas opções da multiseleção")
#     if tipoGrafico == 'Linha':
#         st.line_chart(data=bf, x=selecoes[0], y=selecoes[1])
#     elif tipoGrafico == 'Barra':
#         st.bar_chart(data=bf, x=selecoes[0], y=selecoes[1])
# elif len(selecoes) < 2:
#     st.subheader('Escolha opções na sidebar para formar um gráfico')
# else:
#     st.subheader('Escolha opções para formar um gráfico')
#     st.error("Selecione apenas duas opções")
#
# st.markdown("---")

st.title("Análise da porcentagem da classificação de segurança pela NCAP entre os carros com o seguro ativado")
# Restringindo para apenas aqueles em que o seguro foi ativado
data = bf[bf['is_claim'] == 1]

# Agrupando os valores por ncap_rating e contando a quantidade de ocorrências em cada grupo
count = data.groupby(['ncap_rating'])['is_claim'].count().reset_index(name='count')

# Calculando a porcentagem de ocorrências de cada ncap_rating em relação ao total de is_claim igual a 1
count['Porcentagem'] = count['count'].apply(lambda x: (x/data.shape[0])*100)

# Criando o gráfico de barras
fig = px.bar(count, x='ncap_rating', y='Porcentagem', labels={'ncap_rating':'Classificação de segurança pela NCAP'})
fig.update_traces(text=count['Porcentagem'].apply(lambda x: f'{round(x, 2)}%'))
st.plotly_chart(fig)

st.markdown("---")

st.title("Análise de Taxas de Sinistro por Tipo de Veículo")

# Carregando o dataset
df = pd.read_csv('./data/train.csv')

# Selecionando apenas as colunas relevantes
df = df[['segment', 'is_claim']]

# Agrupando os dados por tipo de veículo e calculando a taxa de sinistro
grouped = df.groupby(['segment']).mean().reset_index()

# Plotando um gráfico de barras
fig = px.bar(grouped, x='segment', y='is_claim', labels={'segment': 'Segmento do carro', 'is_claim': 'Reinvidicado'})
st.write(fig)

st.markdown("---")

st.title("Análise de Probabilidade de Sinistro pela Idade do Motorista")

# Selecionando apenas o grupo relevante
df = bf[bf['is_claim'] == 1][['age_of_policyholder', 'is_claim']]

# Plotando um gráfico violin
fig = px.violin(data_frame=df, y='age_of_policyholder', box=True,
                points='all', labels={'age_of_policyholder': 'Idade do segurado'})

st.write(fig)
st.markdown("---")

st.title("Análise da dispersão entre a idade do segurado e a idade do carro")

# Restringindo para apenas aqueles em que o seguro foi ativado
df = bf[bf['is_claim'] == 1]
# Criando o gráfico junto com a linha de tendência
fig = px.scatter(df, x='age_of_car', y='age_of_policyholder', trendline='ols',
                 labels={'age_of_car': 'Idade do carro', 'age_of_policyholder': 'Idade do segurado'})

# OBS: Se caso não estiver aparecendo o gráfico tenta colocar "pip install statsmodels" no comando e vê se vai
st.write(fig)

# Gráfico Mutual Information Score
st.title("Mutual Information Score")
st.markdown("""<p style="font-size: 16px;text-align: center; margin-top: 0px">
            O Mutual Information Score é uma métrica de aprendizado de máquina que mede a dependência entre duas 
            variáveis aleatórias. No caso do dataset que estamos trabalhando, estamos usando essa métrica para avaliar a
             relação entre as variáveis do dataset e a coluna is_claim.
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
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


y_target = bf.is_claim.astype('int16')

mi_scores = make_mi_scores(bf.drop('is_claim', axis=1), y_target)

# print(mi_scores.head(20))
# print(mi_scores.tail(20))
plt.figure(dpi=100, figsize=(8, 5))
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(plot_mi_scores(mi_scores.head(20)))
