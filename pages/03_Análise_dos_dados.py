import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

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
@st.cache_data
def readData():
    dataset = pd.read_csv('./data/train.csv')
    return dataset


bf = readData()


def tratarDados(database):
    # Apagar coluna policy_id, já que são apenas IDs
    database = database.drop(['policy_id'], axis=1)

    # Tranformar as colunas largura, tamanho e altura em apenas uma coluna chamada volume
    database['volume'] = np.log(database.length * database.width * database.height * 1e-6)

    # Desnormalizando para entender melhor os dados
    database['age_of_policyholder'] = round(database['age_of_policyholder'] * (18 // database['age_of_policyholder'].min()) + database['age_of_policyholder'].min())
    database['age_of_car'] = (database['age_of_car'] * 5)

    # Normalizar policy tenure com min max normalization
    # policy_df = database['policy_tenure']
    # database['policy_tenure'] = (policy_df - policy_df.min()) / (policy_df.max() - policy_df.min())

    database['segment'] = database['segment'].replace('Utility', 'Utilitários')

    database = database.replace({ "No" : False , "Yes" : True,  "Petrol" : "Gasolina" })


    return database


# Criação de array com o nome de todas as colunas para facilitar na criação dos filtros
dict_nome_colunas = ['Idade do carro em anos', 'Idade do segurado', 'Área do segurado',
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
                     'Espaço necessário pro carro fazer uma certa curva', 'Comprimento do carro', 'Largura do carro', 'Altura do carrro'
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
                'turning_radius', 'length', 'width', 'height', 'volume', 'gross_weight', 'is_front_fog_lights', 'is_rear_window_wiper',
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

bf = tratarDados(bf)
numericos = bf.select_dtypes(include=[np.float64, np.int64])
categoricos = bf.select_dtypes(include=[object])

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

# _____________GRÁFICOS USANDO STREAMLIT E PLOTLY________________ #
st.header('Análise da chance de reivindicação')

option = st.selectbox(
    'Seleciona um para comparar com a possibilidade de reivindicação dentro de 6 meses:',
    dict_nome_colunas)

option = dict_nome_colunas.index(option)
data = bf[bf['is_claim'] == 1]
fig = px.bar(data, x=nome_colunas[option], y='is_claim',
             labels={nome_colunas[option]: dict_nome_colunas[option], 'is_claim': 'Reinvidicados'})
st.write(fig)

st.caption('Gráficos para analisar como os valores das colunas interagem com a chance de reivindicação')
st.markdown("---")

# __________________________________________________________________________________________________________________ #

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

# __________________________________________________________________________________________________________________ #

st.subheader(
    "Análise da porcentagem da classificação de segurança pela NCAP entre os carros com chance de reivindicar o seguro")
# Restringindo para apenas aqueles em que o seguro foi ativado
data = bf[bf['is_claim'] == 1]

# Agrupando os valores por ncap_rating e contando a quantidade de ocorrências em cada grupo
count = data.groupby(['ncap_rating'])['is_claim'].count().reset_index(name='count')

# Calculando a porcentagem de ocorrências de cada ncap_rating em relação ao total de is_claim igual a 1
count['Porcentagem'] = count['count'].apply(lambda x: (x / data.shape[0]) * 100)

# Criando o gráfico de barras
fig = px.bar(count, x='ncap_rating', y='Porcentagem', labels={'ncap_rating': 'Classificação de segurança pela NCAP'})
fig.update_traces(text=count['Porcentagem'].apply(lambda x: f'{round(x, 2)}%'))
st.plotly_chart(fig)

st.caption('A análise da porcentagem da classificação de segurança pela NCAP entre os carros com chance de '
           'reivindicar o seguro é importante para as seguradoras avaliarem o grau de segurança dos veículos de maior '
           'risco e estabelecer critérios mais rigorosos para aceitação de seguros. Também incentiva a escolha de '
           'modelos mais seguros pelos clientes, visando a redução dos sinistros e dos preços das apólices.')

st.markdown("---")

# __________________________________________________________________________________________________________________ #

st.subheader("Análise de Probabilidade de Sinistro por Segmento do Veículo")

# Carregando o dataset
# df = pd.read_csv('./data/train.csv')
df = bf

# Selecionando apenas as colunas relevantes
df = df[['segment', 'is_claim']]

# Agrupando os dados por tipo de veículo e calculando a taxa de sinistro
grouped = df.groupby(['segment']).mean().reset_index()

# Plotando um gráfico de barras
fig = px.bar(grouped, x='segment', y='is_claim', labels={'segment': 'Segmento do carro', 'is_claim': 'Reinvidicado'})
st.write(fig)

st.caption('A análise de probabilidade de sinistro por tipo de veículo é importante para as seguradoras, pois permite '
           'avaliar o risco de acidentes envolvendo diferentes tipos de veículos e, consequentemente, estabelecer os '
           'preços das apólices de seguro de forma mais justa e equilibrada. Além disso, essa análise também pode '
           'ajudar a identificar possíveis problemas de segurança em determinados modelos de veículos e, assim, '
           'incentivar melhorias na fabricação de carros mais seguros.')

st.markdown("---")

# __________________________________________________________________________________________________________________ #

# Sugestão: Análise dos veículos com chance de sinistro pela idade do motorista
st.subheader("Análise de Probabilidade de Sinistro pela Idade do Motorista")

# Selecionando apenas o grupo relevante
df = bf[bf['is_claim'] == 1][['age_of_policyholder', 'is_claim']]

# Plotando um gráfico violin
fig = px.violin(data_frame=df, y='age_of_policyholder', box=True,
                points='all', labels={'age_of_policyholder': 'Idade do segurado'})

st.write(fig)

st.caption('A análise de dispersão da probabilidade de sinistro pela idade do motorista é importante para as '
           'seguradoras definirem preços adequados para as apólices de seguro e estabelecer políticas de prevenção de '
           'acidentes. A análise também pode ajudar na criação de programas educacionais para um comportamento mais '
           'seguro no trânsito.')

st.markdown("---")

# __________________________________________________________________________________________________________________ #

st.subheader("Análise da dispersão entre a idade do segurado e a idade do carro")

# Restringindo para apenas aqueles em que o seguro foi ativado
df = bf[bf['is_claim'] == 1]
# Criando o gráfico com a linha de tendência
fig = px.scatter(df, x='age_of_car', y='age_of_policyholder', trendline='ols',
                 labels={'age_of_car': 'Idade do carro', 'age_of_policyholder': 'Idade do segurado'})

# OBS: Caso não estiver a aparecer o gráfico tenta colocar "pip install statsmodels" no comando e vê se vai
st.write(fig)

st.caption('A análise da dispersão entre a idade do segurado e a idade do carro é importante porque permite entender '
           'a relação entre essas variáveis e identificar se há uma tendência de acidentes com carros mais antigos '
           'conduzidos por motoristas mais jovens. Essa análise auxilia na avaliação do risco de sinistros em carros '
           'mais antigos e ajuda a definir políticas mais efetivas de renovação da frota e manutenção preventiva.')

st.markdown("---")

# __________________________________________________________________________________________________________________ #

st.subheader("Análise do tipo de combustível usado pelos carros segurados")

fig = px.histogram(bf, y='fuel_type', labels={'fuel_type': 'Tipo de combustível', 'count': 'Quantidade'})
st.write(fig)

st.caption('Essa análise é importante para as seguradoras entenderem o perfil dos veículos que estão sendo segurados '
           'e as tendências de mercado em relação aos tipos de combustível mais utilizados. Isso pode ajudar a '
           'definir preços mais adequados para as apólices de seguro, bem como a criar políticas e estratégias para '
           'incentivar a adoção de veículos mais sustentáveis e econômicos em termos de combustível.')

st.markdown("---")

# __________________________________________________________________________________________________________________ #

# um histograma que mostre o número de segurados em cada região ou estado.
st.subheader("Análise da distribuição geográfica dos segurados:")

fig = px.histogram(bf, x="area_cluster", labels={'area_cluster': 'Área do Segurado', 'count': 'Quantidade'})
st.write(fig)

st.caption('A análise da distribuição geográfica dos segurados permite que as seguradoras entendam as características '
           'de cada região, como índices de criminalidade e acidentes de trânsito, para ajustar preços e coberturas '
           'de seguro de acordo com o perfil de risco. Também ajuda a desenvolver estratégias de marketing e '
           'atendimento personalizadas para cada localidade.')

st.markdown("---")

# __________________________________________________________________________________________________________________ #

# Um gráfico de dispersão com a densidade populacional no eixo x e a área do segurado no eixo y seria uma boa escolha
# para visualizar a relação entre essas variáveis.
st.subheader("Análise da relação entre a densidade populacional e a área do segurado:")

fig = px.scatter(bf, x='area_cluster', y='population_density',
                 labels={'area_cluster': 'Área do Segurado', 'population_density': 'Densidade Populacional'})
st.write(fig)

st.caption('A análise da relação entre densidade populacional e área do segurado pode ser importante para as '
           'seguradoras ajustarem preços e coberturas de acordo com o perfil de risco de cada área, '
           'além de contribuir para a prevenção de acidentes e roubo de veículos em áreas de maior risco. Essa '
           'análise pode ser útil para entender a demanda por seguros em diferentes áreas geográficas e as possíveis '
           'influências dessas variáveis no risco de sinistro.')

st.markdown("---")

# __________________________________________________________________________________________________________________ #

# Um gráfico de dispersão com a idade do carro no eixo x e o tipo de combustível no eixo y seria uma boa escolha para
# visualizar a relação entre essas variáveis.
st.subheader("Análise da relação entre idade do carro e tipo de combustível:")

fig = px.scatter(bf, x='age_of_car', y='fuel_type',
                 labels={'age_of_car': 'Idade do carro', 'fuel_type': 'Tipo de combustível'})
st.write(fig)

st.caption('Essa análise é importante para entender a relação entre a idade do carro e o tipo de combustível e como '
           'isso afeta o risco de acidentes e o desempenho do veículo. As seguradoras podem usar essas informações '
           'para ajustar preços e coberturas de seguro e incentivar o uso de combustíveis mais eficientes e '
           'sustentáveis')

st.markdown("---")

# __________________________________________________________________________________________________________________ #

# Um gráfico de barras ou um gráfico de pizza pode ser útil para visualizar as frequências de cada recurso de segurança
# para cada classificação de segurança NCAP.
st.subheader("Análise da relação entre a classificação de segurança NCAP e outros recursos de segurança:")

Has_dict_colunas = [
    'Tem controle de estabilização eletrônica?', 'Tem sistema de monitoramento da pressão do pneu?',
    'Tem sensores de ré?',
    'Tem câmera de ré?',
    'Tem farol de neblina?', 'Tem limpador de vidro traseiro?', 'Tem desembaçador de vidro traseiro?',
    'Tem assistência de freio?',
    'Tem trava elétrica de porta?', 'Tem trava central?', 'Tem direção hidráulica?',
    'Tem espelho de retrovisor traseiro?',
    'Tem luz indicativa de problemas no motor?', 'Tem sistema de alerta de velocidade?']
Has_nome_colunas = ['is_esc', 'is_tpms',
                    'is_parking_sensors', 'is_parking_camera', 'is_front_fog_lights',
                    'is_rear_window_washer',
                    'is_rear_window_defogger', 'is_brake_assist', 'is_power_door_locks', 'is_central_locking',
                    'is_power_steering', 'is_day_night_rear_view_mirror',
                    'is_ecw', 'is_speed_alert', 'ncap_rating']

HasOption = st.selectbox(
    'Selecione:',
    Has_dict_colunas)

# Pegando o index para fazer as consultas nas listas criadas acima
HasOption = Has_dict_colunas.index(HasOption)

# Criando um dataset com apenas a classificação e a coluna desejada
dfNcap = pd.concat([bf[Has_nome_colunas[HasOption]], bf['ncap_rating']], axis=1)
dfNcap = dfNcap[dfNcap[Has_nome_colunas[HasOption]] == True]

# Fazendo a contagem
dfNcap_count = dfNcap.groupby(['ncap_rating'])[Has_nome_colunas[HasOption]].count().reset_index(
    name=Has_nome_colunas[HasOption])

fig = px.pie(dfNcap_count, values=Has_nome_colunas[HasOption], names='ncap_rating',
             labels={Has_nome_colunas[HasOption]: Has_dict_colunas[HasOption], 'ncap_rating': 'Classificação de '
                                                                                              'segurança NCAP'},
             color_discrete_sequence=px.colors.qualitative.D3)
fig.update_traces(textinfo='percent+label')

st.write(fig)

st.markdown("""<p style='font-size: 14px;
                    font-weight: 550;
                    margin-top: -50px;
                    color: #9c9d9f;
                    text-align: justify'>
            A análise da relação entre a classificação de segurança NCAP e outros recursos de segurança é importante
           para as seguradoras entenderem como essas variáveis podem influenciar no risco de acidentes e no custo
           das apólices. Isso pode ajudar na definição de preços mais adequados e na criação de políticas de
           prevenção de acidentes de trânsito, considerando fatores como a presença de sistemas de segurança e a
           classificação NCAP do veículo. """, unsafe_allow_html=True)

st.markdown("---")

# __________________________________________________________________________________________________________________ #

# # Botão para página de Machine Learning
# st.markdown("""
#     <br><br><br><br>
#     <div style="text-align: center; margin-top: 60px;">
#     <a href="/Machine_Learning" target="_self"
#     style="text-decoration: none;
#             color: white;
#             font-size: 18px;
#             font-weight: 550;
#             background: rgb(243,68,55);
#             background: linear-gradient(156deg, rgba(243,68,55,1) 30%, rgba(249,170,61,1) 70%);
#             padding: 15px 40px;
#             border-radius: 8px;">
#     Machine Learning
#     </a>
#     </div>
#     """, unsafe_allow_html=True)

# Centralizar todos os elementos da página
st.markdown("""
    <style>
    .element-container {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)