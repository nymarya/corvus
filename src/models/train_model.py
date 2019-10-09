from models import NaiveBayes
import pandas as pd

df = pd.read_csv('data/processed/accidents_dataset.csv', sep=';', 
                 index_col='Unnamed: 0')

df.columns = df.columns.str.replace(' ', '_')

labels_index = df.columns.str.startswith('class_')

nb = NaiveBayes()
nb.train(df, labels_index)
