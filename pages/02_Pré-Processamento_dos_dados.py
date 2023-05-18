import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Tratamento de dados",
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
# @st.cache_data
def readData():
    dataset = pd.read_csv('./data/train.csv')
    return dataset


bf = readData()  # Dataset Tratado

# Criação de array com o nome de todas as colunas para facilitar na criação dos filtros
dict_nome_colunas = ['Idade do carro normalizada', 'Idade do segurado normalizada', 'Área do segurado',
                     'Densidade populacional',
                     'Código da fabricante do carro', 'Segmento do carro (A / B1 / B2 / C1 / C2)', 'Modelo do carro',
                     'Tipo de combustível usado no carro', 'Torque máximo gerado pelo carro',
                     'Força máxima gerada pelo carro',
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

# bf = tratarDados(bf)

st.title('Pré-Processamento dos dados do dataset Car-Insurance')
st.markdown(
    """
    - [Fonte](https://www.kaggle.com/datasets/ifteshanajnin/carinsuranceclaimprediction-classification)
    - [Dicionário de dados](https://github.com/eduardoesnc/SMD/blob/Streamlit/data/Dicionário%20de%20dados.pbf)
    """, unsafe_allow_html=True
)
st.markdown("---")

st.markdown("<h2>Remoção da coluna policyId</h2>", unsafe_allow_html=True)
st.markdown("""<p style="font-size: 16px;text-align: center; margin-top: 0px">
            A coluna 'policy_id' representa apenas os IDs de cada veículo e, portanto, não terá impacto nas análises 
            nem no desenvolvimento da aplicação de machine learning.
            </p>""", unsafe_allow_html=True)

col11, col12 = st.columns(2)
with col11:
    st.markdown("<h5 style='text-align: center;margin-bottom: 0px;'>Antes</h5>", unsafe_allow_html=True)
    st.dataframe(bf)

with col12:
    st.markdown("<h5 style='text-align: center;margin-bottom: 0px;'>Depois</h5>", unsafe_allow_html=True)
    bf = bf.drop(['policy_id'], axis=1)
    st.dataframe(bf)

st.markdown("---")

# ______________________________________________Idade do Segurado____________________________________________________________________ #

st.markdown("<h2>Desnormalização da Idade do Segurado</h2>", unsafe_allow_html=True)
st.markdown("""<p style="font-size: 16px;text-align: center; margin-top: 0px">
            A coluna que indica a Idade do Segurado será desnormalizada para permitir uma melhor compreensão dos dados,
             ajustando-a considerando idades de 0 a 65 anos (valores definidos considerando que a menor idade no dataset representa 18 anos).
             Vale lembrar que, para utilizar essa coluna em um algoritmo de Machine Learning, utilizaremos a versão
             normalizada dela. Ao fazer isso, teremos uma visão mais abrangente e precisa dos dados, o que permitirá
             uma análise mais eficiente e tomada de decisões mais informadas.
            </p>""", unsafe_allow_html=True)
A1, A2 = st.columns(2)
with A1:
    st.markdown("<h5 style='text-align: center;margin-bottom: 0px;'>Normalizada</h5>", unsafe_allow_html=True)
    columnsPT = bf['age_of_policyholder']
    st.dataframe(columnsPT)

with A2:
    st.markdown("<h5 style='text-align: center;margin-bottom: 0px;'>Desnormalizada</h5>", unsafe_allow_html=True)
    bf['age_of_policyholder'] = round(bf['age_of_policyholder'] * (18 // bf['age_of_policyholder'].min()) + bf['age_of_policyholder'].min())
    st.dataframe(bf['age_of_policyholder'])

st.markdown("---")

# ______________________________________________Idade do Carro____________________________________________________________________ #

st.markdown("<h2>Desnormalização da Idade do Carro</h2>", unsafe_allow_html=True)
st.markdown("""<p style="font-size: 16px;text-align: center; margin-top: 0px">
            A coluna que indica a Idade do Carro será desnormalizada para permitir uma melhor compreensão dos dados, ajustando-a
             considerando idades de 0 a 5 anos (valores definidos considerando a vida útil média de um carro). Vale lembrar que,
             para utilizar essa coluna em um algoritmo de Machine Learning, utilizaremos a versão normalizada dela. Ao fazer isso,
             teremos uma visão mais abrangente e precisa dos dados, o que permitirá uma análise mais eficiente e tomada de decisões
             mais informadas.
            </p>""", unsafe_allow_html=True)
A1, A2 = st.columns(2)
with A1:
    st.markdown("<h5 style='text-align: center;margin-bottom: 0px;'>Normalizada</h5>", unsafe_allow_html=True)
    columnsPT = bf['age_of_car']
    st.dataframe(columnsPT)

with A2:
    st.markdown("<h5 style='text-align: center;margin-bottom: 0px;'>Desnormalizada</h5>", unsafe_allow_html=True)
    bf['age_of_car'] = (bf['age_of_car'] * 5)
    st.dataframe(bf['age_of_car'])

st.markdown("---")

# __________________________________________________________________________________________________________________ #

# IREMOS USAR O COMPRIMENTO MESMO

# st.markdown("<h2>Altura, largura e comprimento para volume</h2>", unsafe_allow_html=True)
# st.markdown("""<p style="font-size: 16px;text-align: center; margin-top: 0px">
#             Pode-se simplificar as colunas de altura, largura e comprimento em uma única coluna, denominada 'Volume'. É 
#             importante normalizar a coluna Volume para que seus valores estejam em uma escala comum e, assim, possamos 
#             trabalhar com mais eficiência no algoritmo de Machine Learning.
#             </p>""", unsafe_allow_html=True)

# B1, B2, B3 = st.columns(3)
# with B1:
#     st.markdown("<h5 style='text-align: center;margin-bottom: 0px;'>Original</h5>", unsafe_allow_html=True)
#     columnsALC = pd.concat([bf['length'], bf['width'], bf['height']], axis=1)
#     st.dataframe(columnsALC)

# with B2:
#     # Transformando em apenas uma coluna
#     st.markdown("<h5 style='text-align: center;margin-bottom: 0px;'>Transformada</h5>", unsafe_allow_html=True)
#     # Os valores estavam em milímetros
#     bf['volume'] = np.log(bf.length * bf.width * bf.height * 1e-6)
#     bf = bf.drop(['length', 'width', 'height'], axis=1)
#     columnVol = bf['volume']
#     st.dataframe(columnVol)

# with B3:
#     # Padronizandos os valores
#     st.markdown("<h5 style='text-align: center;margin-bottom: 0px;'>Normalizada</h5>", unsafe_allow_html=True)
#     normColumnVol = columnVol
#     normColumnVol = (normColumnVol - normColumnVol.min()) / (normColumnVol.max() - normColumnVol.min())
#     normColumnVol = pd.concat([normColumnVol], axis=1)
#     st.dataframe(normColumnVol)

# st.markdown("---")

# __________________________________________________________________________________________________________________ #

st.markdown("<h2>Outliers</h2>", unsafe_allow_html=True)
st.markdown("""<p style="font-size: 14px;font-style: italic;color: gray; margin-top: 
    -15px">Outliers são valores atípicos ou extremos em uma série de dados que podem afetar a análise e precisam ser 
    identificados e tratados antes da análise estatística.</p>""", unsafe_allow_html=True)

# Filtragens do gráfico usado apenas para a busca de Outliers
# st.markdown("<br> <h4 style='text-align: center;'>Busca por Outliers</h4>", unsafe_allow_html=True)
# option = st.selectbox(
#     'Seleciona um para comparar com a possibilidade de reivindicação dentro de 6 meses:',
#     dict_nome_colunas)
# option = dict_nome_colunas.index(option)
# fig = px.box(bf, x=nome_colunas[option], labels={nome_colunas[option]: dict_nome_colunas[option]}, height=300)
# st.write(fig)

st.markdown("<br> <h4 style='text-align: center;'>Idade do carro</h4>", unsafe_allow_html=True)
C1, C2 = st.columns(2)

with C1:
    st.markdown("<h5 style='text-align: center;margin-bottom: -34px;'>Antes</h5>", unsafe_allow_html=True)
    fig = px.box(bf, x='age_of_car', labels={'age_of_car': 'Idade do carro'}, width=400, height=300)
    st.write(fig)

with C2:
    st.markdown("<h5 style='text-align: center; margin-bottom: -34px;'>Depois</h5>", unsafe_allow_html=True)
    age_of_car_outliers = bf.age_of_car > bf.age_of_car.quantile(0.995)
    bf = bf.loc[~age_of_car_outliers]

    fig = px.box(bf, x='age_of_car', labels={'age_of_car': 'Idade do carro'}, width=400, height=300)
    st.write(fig)

st.markdown("<h4 style='text-align: center;'>Idade do segurado</h4>", unsafe_allow_html=True)
D1, D2 = st.columns(2)

with D1:
    st.markdown("<h5 style='text-align: center;margin-bottom: -34px;'>Antes</h5>", unsafe_allow_html=True)
    fig = px.box(bf, x='age_of_policyholder', labels={'age_of_policyholder': 'Idade do segurado'},
                 width=400, height=300)
    st.write(fig)

with D2:
    st.markdown("<h5 style='text-align: center; margin-bottom: -34px;'>Depois</h5>", unsafe_allow_html=True)
    age_of_policyholder_outliers = bf.age_of_policyholder > bf.age_of_policyholder.quantile(0.995)
    bf = bf.loc[~age_of_policyholder_outliers]

    fig = px.box(bf, x='age_of_policyholder', labels={'age_of_policyholder': 'Idade do segurado'},
                 width=400, height=300)
    st.write(fig)

st.markdown("""<p style="font-size: 16px;text-align: center; margin-top: -20px">
            As colunas da idade do carro e da idade do segurado apontavam alguns valores muito diferentes do comum, 
            sendo assim foram removidas cerca de 269 linhas que apresentavam esses valores extremos.
            </p>""", unsafe_allow_html=True)

st.markdown("---")
# __________________________________________________________________________________________________________________ #

# Botão para página de análise de dados
st.markdown("""
    <br><br>
    <div style="text-align: center; margin-top: 60px;">
    <a href="/Análise_dos_dados" target="_self"
    style="text-decoration: none;
            color: white;
            font-size: 18px;
            font-weight: 550;
            background: rgb(243,68,55);
            background: linear-gradient(156deg, rgba(243,68,55,1) 30%, rgba(249,170,61,1) 70%);
            padding: 15px 48px;
            border-radius: 8px;">
    Análise exploratória dos dados
    </a>
    </div>
    """, unsafe_allow_html=True)

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
