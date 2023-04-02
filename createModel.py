import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

def readData():
    dataset = pd.read_csv('./data/train.csv')
    datasetTreino = pd.read_csv('./data/test.csv')
    sampleSub = pd.read_csv('./data/sample_submission.csv')
    return dataset, datasetTreino, sampleSub

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

df = tratarDados(df)

# Criando dataframe temporário apenas com as colunas selecionadas
dfRF = df.copy()
dfPlusRF = dfPlus.copy()

sample = sample.drop(['policy_id'], axis=1)

dfPlusRF = pd.concat([pd.DataFrame(dfPlusRF), pd.DataFrame(sample)], axis=1)

dfPlusRF = tratarDados(dfPlusRF)

dfRF = pd.concat([pd.DataFrame(dfRF), pd.DataFrame(dfPlusRF)], axis = 0)

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
x_train, x_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2)

# Inicializando o algoritmo Random Forest
classifier = RandomForestClassifier()

#  Construindo uma 'floreste de árvores' da parte de treino
classifier.fit(x_train,y_train)

with open("model.pkl", "wb") as f:
     pickle.dump(classifier, f)