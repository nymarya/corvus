from models import NaiveBayes, SVM
import pandas as pd

import argparse
import pickle
from datetime import datetime

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('model', metavar='string', type=str, nargs='?',
                    help='')

args = parser.parse_args()
model = args.model

df = pd.read_csv('data/processed/accidents_dataset.csv', sep=';',
                 index_col='Unnamed: 0')

df.columns = df.columns.str.replace(' ', '_')

labels_index = df.columns.str.startswith('classificacao')

# Split data
df_copy = df.copy()
train_set = df_copy.sample(frac=0.90, random_state=42)
test_set = df_copy.drop(train_set.index)
test_labels = test_set.columns[labels_index]
test_set.drop(labels=df.columns[labels_index], axis=1, inplace=True)
assert(test_set.shape[1] == (df.shape[1]-1))

if(model == 'naive_bayes'):
    trained = NaiveBayes()
    trained.train(train_set, labels_index)

    trained.test(test_set, test_labels)

    print("Trained")
elif (model == "svm"):
    trained = SVM()
    trained.train(train_set, labels_index)
else:
    print("Invalid type: " + model)

# Serialize model using pickle
date = datetime.now().strftime("%Y%m%d_%H%M")
filename = 'models/{}_{}.pickle'.format(model, date)
with open(filename, 'wb') as f:
    pickle.dump(trained, f, pickle.HIGHEST_PROTOCOL)
    print("Model save at: " + filename)
