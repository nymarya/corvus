from models import NaiveBayes
import pandas as pd

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('model', metavar='string', type=str, nargs='?',
                    help='')

args = parser.parse_args()
model = args.model

df = pd.read_csv('data/processed/accidents_dataset.csv', sep=';', 
                 index_col='Unnamed: 0')

df.columns = df.columns.str.replace(' ', '_')

labels_index = df.columns.str.startswith('class_')

# Split data
df_copy = df.copy()
train_set = df_copy.sample(frac=0.77, random_state=42)
test_set = df_copy.drop(train_set.index)
test_set.drop(labels=df.columns[labels_index], axis=1, inplace=True)
assert(test_set.shape[1] == (df.shape[1]-4))

if(model == 'naive_bayes'):
    nb = NaiveBayes()
    nb.train(train_set, labels_index)

    print("Trained")
else:
    print("Invalid type: " + model)
