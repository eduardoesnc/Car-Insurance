import streamlit as st
import pickle

st.set_page_config(
    page_title="Estimar",
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

st.markdown("""<h1 style='text-align: center; margin-top: -60px'>Realizar Estimativa</h1> <br>""", unsafe_allow_html=True)

# comprimento, Tempo de seguro, Idade do carro, Área do segurado, Idade do segurado, Modelo.

comprimento = st.number_input('Comprimento em mm:')

tempoSeguro = st.number_input('Tempo de Seguro:')

idadeCarro = st.number_input('Idade do Carro:')

idadeSegurado = st.number_input('Idade do Segurado:')

areaSeguradoList = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12',
                    'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22']

areaSegurado = st.selectbox('Área do Segurado', areaSeguradoList)

modeloCarroList = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11']

modeloCarro = st.selectbox('Modelo do Carro', modeloCarroList)

def make_prediction(comprimento,tempoSeguro, idadeCarro, 
                    idadeSegurado, areaSegurado, modeloCarro):
    count = 0
    for i in areaSeguradoList:
        if i == areaSegurado:
            areaSeguradoInt = count
        count += 1
    
    count = 0
    for i in modeloCarroList:
        if i == modeloCarro:
            modeloCarroInt = count
        count += 1

    with open("model.pkl", "rb") as f:
        clf  = pickle.load(f)
        preds = clf.predict([[comprimento,tempoSeguro, idadeCarro, 
                    idadeSegurado, areaSeguradoInt, modeloCarroInt]])
    return preds

st.markdown("""<br>""", unsafe_allow_html=True)

if st.button("Estimar"):
    results = make_prediction(comprimento,tempoSeguro, idadeCarro, idadeSegurado, areaSegurado, modeloCarro)
    st.markdown("""<br>""", unsafe_allow_html=True)
    if results == 1:
        st.error("Grande chance de reivindicar o seguro")
    else:
        st.success("Pouca chance de reivindicar o seguro")
        
st.markdown("""<br>""", unsafe_allow_html=True)
st.caption('Apenas realizamos estimativas')


# PARA CENTRALIZAR OS GRÁFICOS E TABELAS NA PÁGINA (MANTER SEMPRE NO FINAL DO ARQUIVO)
st.markdown("""
   <style>
   .element-container {
       display: flex;
       justify-content: center;
       align-items: center;
   }

   .image {
       display: flex;
       justify-content: center;
       align-items: center;
   }
   </style>
   """, unsafe_allow_html=True)