import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(
    page_title = "Tratamento de dados",
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

# Leitura e tratamento dos dados
# @st.cache
def readData():
    dataset = pd.read_csv('./data/train.csv')
    return dataset
bf = readData()

def tratarDados(df):
    # Voltando idade do segurado para o normal
    df['age_of_policyholder'] = round(df['age_of_policyholder'].mul(100))

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

bf = tratarDados(bf) # tbf é o dataset com tratamento de dados
numericos = bf.select_dtypes(include=[np.float64, np.int64])
categoricos = bf.select_dtypes(include=[np.object])

st.title('Tratamento dos dados do dataset Car-Insurance')
st.markdown(
    """
    - [Fonte](https://www.kaggle.com/datasets/ifteshanajnin/carinsuranceclaimprediction-classification)
    - [Dicionário de dados](https://github.com/eduardoesnc/SMD/blob/streamlit/data/Dicionário%20de%20dados%20-%20Car%20Insurance%20Database.pdf)
    """, unsafe_allow_html=True
)
st.markdown("---")

st.subheader('Outliers')
st.markdown("""<p style="font-size: 14px;font-style: italic;color: gray;text-align: center; margin-top: -20px">Outliers são valores atípicos ou extremos em uma série de dados que podem afetar a análise e precisam ser
            identificados e tratados antes da análise estatística.</p>""", unsafe_allow_html=True)


st.markdown("<br> <h5 style='text-align: center;'>Idade do carro (age_of_car)</h5>", unsafe_allow_html=True)
A1, A2 = st.columns(2)

with A1:
    st.markdown("<h5 style='text-align: center;margin-bottom: -34px;'>Antes</h5>", unsafe_allow_html=True)
    fig = px.box(bf, x='age_of_car')
    st.write(fig)

with A2:
    st.markdown("<h5 style='text-align: center; margin-bottom: -34px;'>Depois</h5>", unsafe_allow_html=True)
    age_of_car_outliers = bf.age_of_car > bf.age_of_car.quantile(0.5)
    bf = bf.loc[~age_of_car_outliers]

    fig = px.box(bf, x='age_of_car')
    st.write(fig)

st.markdown("""<p style="font-size: 16px;text-align: center; margin-top: -20px">
            A coluna ´age_of_car´ apresentava alguns valores muito diferentes do comum, sendo assim foram removidas
            algumas linhas que apresentavam esses valores extremos.
            </p>""", unsafe_allow_html=True)
st.markdown("---")