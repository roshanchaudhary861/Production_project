import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    df['ActivityDate'] = pd.to_datetime(df['ActivityDate'])
    df = df.drop_duplicates()
    df = df.dropna()
    return df