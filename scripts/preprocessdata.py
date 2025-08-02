import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pickle
from src.utils.preprocessing import clean_text


def preprocess_data():
    df = pd.read_csv('data\ActualData\IMDB Dataset.csv')
    print(df.head())
    df['processed_review'] = df['review'].apply(clean_text)
    df = df[['processed_review', 'sentiment']]
    print('*'*50)
    print('*'*50)
    print(df.head())
    df.to_pickle('data/PreprocessedData/preprocessed_data.pkl')
    print("saved Done")