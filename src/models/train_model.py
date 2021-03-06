from models import NaiveBayes, KNN
import pandas as pd

import argparse
import pickle
from datetime import datetime

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('model', metavar='string', type=str, nargs='?',
                    help='')
parser.add_argument('--metric', metavar='string', type=str, nargs='?',
                    help='')
parser.add_argument('--k', metavar='string', type=int, nargs='?',
                    help='')

args = parser.parse_args()
model = args.model
metric = args.metric
k = args.k

df = pd.read_csv('data/processed/accidents_dataset.csv', sep=';',
                 index_col='Unnamed: 0')

df.columns = df.columns.str.replace(' ', '_')

# Split data
df_copy = df.copy()
train_set = df_copy.sample(frac=0.77, random_state=42)
test_set = df_copy.drop(train_set.index)
test_labels = test_set['classificacao_acidente']
test_set.drop(labels=['classificacao_acidente'], axis=1, inplace=True)
assert(test_set.shape[1] == (df.shape[1]-1))

if(model == 'naive_bayes'):
    trained = NaiveBayes()
# elif (model == "svm"):
#     trained = SVM()
#     trained.train(train_set, 'classificacao_acidente')
elif (model == "knn"):
    if(metric is not None):
        trained = KNN(k, metric)
    else:
        trained = KNN(k)

else:
    print("Invalid type: " + model)

trained.train(train_set, 'classificacao_acidente')

trained.test(test_set, test_labels)

print(trained._accuracy())

print("Trained")
# Serialize model using pickle
date = datetime.now().strftime("%Y%m%d_%H%M")
filename = 'models/{}_{}.pickle'.format(model, date)
with open(filename, 'wb') as f:
    pickle.dump(trained, f, pickle.HIGHEST_PROTOCOL)
    print("Model save at: " + filename)
