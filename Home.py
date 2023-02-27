import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Home",
    page_icon="assets/car-logo.png",
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


bf = readData()

# def tratarDados(df):
# Idade do segurado
# df['age_of_policyholder'] = round(df['age_of_policyholder'].mul(100))
# max_torque e max_power
# df["max_torque_Nm"] = df['max_torque'].str.extract(r"([-+]?[0-9]*\.?[0-9]+)(?=\s*Nm)").astype('float64')
# df["max_torque_rpm"] = df['max_torque'].str.extract(r"([-+]?[0-9]*\.?[0-9]+)(?=\s*rpm)").astype('float64')
#
# df["max_power_bhp"] = df['max_power'].str.extract(r"([-+]?[0-9]*\.?[0-9]+)(?=\s*bhp)").astype('float64')
# df["max_power_rpm"] = df['max_power'].str.extract(r"([-+]?[0-9]*\.?[0-9]+)(?=\s*rpm)").astype('float64')


# tratarDados(bf)

# st.sidebar.image('./assets/logo InsuranceTech.png', caption='InsuranceTech', use_column_width=True)
# st.sidebar.header('Dashboard')

st.title('Bem-vindo!')

st.markdown(
    """
    O conjunto de dados [Car Insurance Claim
     Prediction](https://www.kaggle.com/datasets/ifteshanajnin/carinsuranceclaimprediction-classification) contém 
     informações sobre clientes de uma seguradora de automóveis, incluindo detalhes sobre seus veículos, histórico 
     de sinistros e outras informações pessoais. O objetivo deste dashboard é fornecer uma visão geral desses dados,
      permitindo que os usuários explorem e analisem as informações relevantes para desenvolver uma aplicação de Machine
       Learning para prever se um cliente fará uma reclamação de seguro.

    Com base nas análises realizadas neste dashboard, os usuários podem identificar quais variáveis estão mais 
    correlacionadas com as reclamações de seguros e quais técnicas de Machine Learning são mais adequadas para 
    desenvolver um modelo de previsão preciso. Além disso, este dashboard pode ajudar os usuários a identificar 
    quais clientes são mais propensos a fazer uma reclamação de seguro, permitindo que a seguradora tome medidas 
    proativas para reduzir o risco de sinistros e aumentar a satisfação do cliente.

    A integração deste dashboard com uma aplicação Flutter pode permitir que os usuários acessem as informações e 
    visualizações de dados em tempo real a partir de seus dispositivos móveis, tornando a análise de dados mais 
    acessível e conveniente. Combinado com o poder da Machine Learning, essa aplicação pode ajudar a seguradora 
    a melhorar seus serviços e tomar decisões mais informadas e precisas.

    #### Quer saber mais?
    - [Artigo do projeto](https://docs.google.com/document/d/1i_8mR6b4knryxF9xFbUrW3NNg7GUYP-RoosVgll9K1k/edit?usp=sharing)
    - [Github do projeto](https://github.com/eduardoesnc/SMD)
    
    <h2 style='text-align: center;'> Informações básicas do dataset </h2>
    <br>
    <br>
    """, unsafe_allow_html=True
)

# Vamos chamar as linhas pelas letras do alfabeto e as colunas por números
A1, A2 = st.columns(2)

with A1:
    st.markdown("<h4 style='text-align: center;'>Dataset</h4>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(bf)

with A2:
    st.markdown("<h4 style='text-align: center;'>Tipos dos dados</h4>", unsafe_allow_html=True)
    st.caption("<a href='https://github.com/eduardoesnc/SMD/blob/Streamlit/data/Dicionário%20de%20dados.pdf'>"
               "<p style='text-align: center;margin: -5%;'> Dicionário de dados </p>"
               "</a>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    bf.dtypes
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

B1, B2, B3 = st.columns(3)

with B1:
    st.markdown("<h5 style='text-align: center;'>Composto por</h5>", unsafe_allow_html=True)
    st.markdown("""
            <div style='text-align: center;'>
                • 58592 linhas
                <br>
                • 44 colunas
            </div>
            """, unsafe_allow_html=True)

with B2:
    st.markdown("<h5 style='text-align: center;'>Tipos de recursos</h5>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center;'>
        • 17 recursos categóricos
        <br>
        • 10 recursos numéricos
        <br>
        • 17 recursos booleanos
    </div>
    """, unsafe_allow_html=True)

with B3:
    isClaimTrue = sum(bf.is_claim)
    isClaimFalse = len(bf.is_claim) - isClaimTrue
    razaoIsClaim = round(isClaimFalse / isClaimTrue, 1)
    # Para o dataset train.csv essa razão é de 1 True para 14.6 False

    st.markdown("<h5 style='text-align: center;'>Razão isClaim</h5>", unsafe_allow_html=True)
    st.metric('Razão da coluna isClaim', '1/14.6', delta=None, delta_color="normal",
              help="Temos 14.6 isClaim falsos para cada isClaim verdadeiro", )
    st.markdown("<br> <br>", unsafe_allow_html=True)

C1, C2, C3 = st.columns(3)

with C1:
    # print(bf.isnull().sum())
    st.markdown("<h5 style='text-align: center;'>Quantidade de valores nulos</h5>", unsafe_allow_html=True)
    st.metric('Quantidade de valores nulos', 0, delta=None, delta_color="normal",
              help="O dataset não apresenta valores nulos", )

with C2:
    qtdDuplicados = bf.duplicated().sum()
    st.markdown("<h5 style='text-align: center;'>Quantidade de valores duplicados</h5>", unsafe_allow_html=True)
    st.metric('Quantidade de valores duplicados', qtdDuplicados, delta=None, delta_color="normal",
              help="O dataset não apresenta valores duplicados")

with C3:
    st.markdown("<h5 style='text-align: center;'>Quantidade de colunas multivaloradas</h5>", unsafe_allow_html=True)
    st.metric('Quantidade de colunas multivaloradas', 0, delta=None, delta_color="normal",
              help="O dataset não apresenta colunas multivaloradas", )

st.markdown("""
    <div style="text-align: center; margin-top: 60px;">
    <a href="/Pré-Processamento_dos_dados" target="_self"
    style="text-decoration: none;
            color: white;
            font-size: 18px;
            font-weight: 550;
            background: rgb(243,68,55);
            background: linear-gradient(156deg, rgba(249,170,61,1) 30%, rgba(243,68,55,1) 70%);
            padding: 15px 40px;
            border-radius: 8px;">
    Pré-Processamento dos dados
    </a>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; margin-top: 60px;">
    <a href="/Análise_dos_dados" target="_self"
    style="text-decoration: none;
            color: white;
            font-size: 18px;
            font-weight: 550;
            background: rgb(243,68,55);
            background: linear-gradient(156deg, rgba(243,68,55,1) 30%, rgba(249,170,61,1) 70%);
            padding: 15px 40px;
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

