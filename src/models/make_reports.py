import matplotlib.pyplot as plt
import numpy as np
import pickle
from datetime import datetime
import argparse
import os

def plot_confusion_matrix(model, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, metrics=None, labels=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    fig, ax = plt.subplots()
    # We want to show all ticks...
    xlabel = 'Predicted label'
    if(metrics is not None):
        xlabel += '\n{}'.format(metrics)

    ax.set(xticks=np.arange(4),
           yticks=np.arange(4),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel=xlabel)
    im = ax.imshow(cm, cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.align_xlabels()
    plt.tight_layout()
    filename = 'reports/figures/{}.png'.format(model)
    plt.savefig(filename)

# Receive argument
parser = argparse.ArgumentParser(description='Recove models.')
parser.add_argument('files', metavar='string', type=str, nargs='?',
                    help='')

args = parser.parse_args()
substr = args.files

for filename in os.listdir('models'):
    # unpickle
    model = os.path.join("models", filename)
    # filter files
    if substr not in filename:
        continue
    with open(model, 'rb') as f:
        name = filename.split('.')[0]
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        model = pickle.load(f)
        report = model.report()
        labels = ['FATAL_VICTIMS', 'INJURED_VICTIMS', 'IGNORED', 'NO_VICTIMS']
        confusion_matrix = np.matrix(model.confusion_matrix['matrix'])
        plot_confusion_matrix(name, confusion_matrix, classes=labels,
                              metrics=report,
                              title='Confusion matrix\n'+model.to_string())

        # plt.show()
