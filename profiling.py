import pandas as pd
from pandas_profiling import ProfileReport
from pandas_profiling.utils.cache import cache_file

def readData():
    dataset = pd.read_csv('./data/train.csv')
    # Normalizar valores da idade
    dataset['age_of_policyholder'] = round(dataset['age_of_policyholder'].mul(100))
    return dataset
bf = readData()

profile = ProfileReport(bf, title=f"Car Insurance Dataset")
profile.to_file(f"car-insurance.html")