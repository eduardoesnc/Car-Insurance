import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px



st.set_page_config(
    page_title = "EDA",
    page_icon = './assets/car-logo.png',
    layout = "wide",
    menu_items = {
        'About': "Desenvolvedores: "
                 "Anna Carolina Lopes Dias Coêlho Bejan, "
                 "Atlas Cipriano da Silva, "
                 "Eduardo Estevão Nunes Cavalcante, "
                 "Matheus Mota Fernandes"
    },
    initial_sidebar_state='expanded'
)

st.sidebar.header('Dashboard')

# Leitura e tratamento dos dados
# @st.cache
def readData():
    dataset = pd.read_csv('./data/train.csv')
    return dataset
bf = readData()


def tratarDados(df):
    # Voltando idade do segurado para o normal
    df['age_of_policyholder'] = round(df['age_of_policyholder'].mul(100))
    df['age_of_car'] = round(df['age_of_car'].mul(100))

    #Separando as tabelas max_torque e max_power
    df["max_torque_Nm"] = df['max_torque'].str.extract(r"([-+]?[0-9]*\.?[0-9]+)(?=\s*Nm)").astype('float64')
    df["max_torque_rpm"] = df['max_torque'].str.extract(r"([-+]?[0-9]*\.?[0-9]+)(?=\s*rpm)").astype('float64')

    df["max_power_bhp"] = df['max_power'].str.extract(r"([-+]?[0-9]*\.?[0-9]+)(?=\s*bhp)").astype('float64')
    df["max_power_rpm"] = df['max_power'].str.extract(r"([-+]?[0-9]*\.?[0-9]+)(?=\s*rpm)").astype('float64')

    df = df.drop(['max_torque'], axis=1)
    df = df.drop(['max_power'], axis=1)

    #Apagar coluna policy_id, já que são apenas IDs
    df = df.drop(['policy_id'], axis=1)

    #Tranformar as colunas largura tamanho e altura em apenas uma coluna chamada volume
    df['volume'] = np.log(df.length * df.width * df.height * 1e-6)
    df = df.drop(['length', 'width', 'height'], axis=1)

    # Normalizar policy tenure com min max normalization
    # policy_df = bf['policy_tenure']
    # normPolicy = (policy_df - policy_df.min()) / (policy_df.max() - policy_df.min())
    # normPolicy = pd.concat([normPolicy, bf['is_claim']], axis=1)

    return df




#Criação de array com o nome de todas as colunas para facilitar na criação dos filtros
dict_nome_colunas = ['Idade do carro em anos', 'Idade do segurado em anos', 'Área do segurado', 'Densidade populacional',
     'Código da fabricante do carro', 'Segmento do carro (A / B1 / B2 / C1 / C2)', 'Modelo do carro',
     'Tipo de combustível usado no carro', 'Torque máximo gerado pelo carro (Nm@rpm)', 'Força máxima gerada pelo carro (bhp@rpm)',
     'Tipo de motor usado pelo carro', 'Número de airbags instalados no carro', 'Tem controle de estabilização eletrônica?',
     'O volante é ajustável?', 'Tem sistema de monitoramento da pressão do pneu?', 'Tem sensores de ré?',
     'Tem câmera de ré?', 'Tipo de freio usado no carro', 'Cilindradas do motor (cc)', 'Quantidade de cilindros do carro',
     'Tipo de transmissão do carro', 'Quantidade de marchas do carro', 'Tipo de direção do carro', 'Espaço necessário pro carro fazer uma certa curva',
     'Volume do carro', 'Peso máximo suportado pelo carro',
     'Tem farol de neblina?', 'Tem limpador de vidro traseiro?', 'Tem desembaçador de vidro traseiro?', 'Tem assistência de freio?',
     'Tem trava elétrica de porta?', 'Tem direção hidráulica?', 'O acento do motorista é ajustável?', 'Tem espelho de retrovisor traseiro?',
     'Tem luz indicativa de problemas no motor?', 'Tem sistema de alerta de velocidade?', 'Classificação de segurança pela NCAP (de 0 a 5)']
nome_colunas = ['age_of_car','age_of_policyholder','area_cluster','population_density','make','segment'
    ,'model','fuel_type','max_torque_Nm','max_power_bhp','engine_type','airbags','is_esc','is_adjustable_steering','is_tpms',
    'is_parking_sensors','is_parking_camera','rear_brakes_type','displacement','cylinder','transmission_type','gear_box','steering_type',
    'turning_radius','volume','gross_weight','is_front_fog_lights','is_rear_window_wiper','is_rear_window_washer'
    ,'is_rear_window_defogger','is_brake_assist','is_power_door_locks','is_central_locking,is_power_steering',
    'is_driver_seat_height_adjustable','is_day_night_rear_view_mirror','is_ecw','is_speed_alert','ncap_rating','is_claim']

# Verificando se há valores nulos
@st.cache
def manterDados(bf):
    null = bf.isnull().sum()
    return null

st.title('Ánalise de dados com o dataset Car-Insurance')

#Download do dataset
@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')

csv = convert_df(bf)

st.download_button(
    label="Baixar dataset",
    data=csv,
    file_name='car-insurance.csv',
    mime='text/csv',
)

bf = tratarDados(bf) # tbf é o dataset com tratamento de dados
numericos = bf.select_dtypes(include=[np.float64, np.int64])
categoricos = bf.select_dtypes(include=[np.object])

#___________________________________________________#

st.markdown(
    """
    - [Fonte](https://www.kaggle.com/datasets/ifteshanajnin/carinsuranceclaimprediction-classification)
    - [Dicionário de dados](https://github.com/eduardoesnc/SMD/blob/streamlit/data/Dicionário%20de%20dados%20-%20Car%20Insurance%20Database.pdf)
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
if st.sidebar.checkbox('Mostrar os tipos dos dados do dataset'):
    st.subheader('Tipos dos dados')
    bf.dtypes

st.sidebar.markdown("---")

# _____________GRÁFICOS USANDO STREAMLIT E PLOTLY________________ #
st.header('Análises')

option = st.selectbox(
    'Seleciona um para comparar com a possibilidade de reivindicação dentro de 6 meses:',
    dict_nome_colunas)

option = dict_nome_colunas.index(option)

st.bar_chart(data=bf, x=nome_colunas[option], y='is_claim')


st.caption('Gráficos para analisar como os valores das colunas interagem com a chance de reivindicação')
st.markdown("---")


selecoes = st.sidebar.multiselect('Escolha duas opções e o tipo de gráfico desejado', nome_colunas)
tipoGrafico = st.sidebar.radio('',['Linha', 'Barra'])
if len(selecoes) == 2:
    st.subheader("Gráficos formados pelas opções da multiseleção")
    if tipoGrafico == 'Linha':
        st.line_chart(data=bf, x=selecoes[0], y=selecoes[1])
    elif tipoGrafico == 'Barra':
        st.bar_chart(data=bf, x=selecoes[0], y=selecoes[1])
elif len(selecoes) < 2:
    st.subheader('Escolha opções na sidebar para formar um gráfico')
else:
    st.subheader('Escolha opções para formar um gráfico')
    st.error("Selecione apenas duas opções")

st.markdown("---")


st.title("Análise de Taxas de Sinistro por Tipo de Veículo")

# Carregando o dataset
df = pd.read_csv('./data/train.csv')

# Selecionando apenas as colunas relevantes
df = df[['segment', 'is_claim']]

# Agrupando os dados por tipo de veículo e calculando a taxa de sinistro
grouped = df.groupby(['segment']).mean().reset_index()

# Plotando um gráfico de barras
st.bar_chart(data=grouped, y='segment')
st.markdown("---")

st.title("Análise de Probabilidade de Sinistro por Tempo de Carteira")

# Carregando o dataset
df = pd.read_csv('./data/train.csv')

# Selecionando apenas o grupo relevante
df = bf[bf['is_claim'] == 1][['age_of_policyholder', 'is_claim']]

# Plotando um gráfico violin
fig = px.violin(data_frame=df, y='age_of_policyholder', box=True, 
               points='all')

st.write(fig)
st.markdown("---")

st.title("Análise da dispersão entre a idade do segurado e a idade do carro")

# Carregando o dataset
df = pd.read_csv('./data/train.csv')
# Restringindo para apenas aqueles em que o seguro foi ativado
df = bf[bf['is_claim'] == 1]
#Criando o gráfico junto com a linha de tendência
fig = px.scatter(df, x='age_of_car', y='age_of_policyholder', trendline='ols')

st.write(fig)
