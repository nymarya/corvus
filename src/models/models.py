import pandas as pd
import numpy as np
from model import Model
import operator


class NaiveBayes(Model):
    """ Implementation of a naïve version of the Bayesian belief network.
    The features are assumed to be independent.

    P(B | A) = P( B ^ A) / P(A)
    """

    # Keep the probabilities of each class as a dictionary.
    # E.g.: {'A': 0.5, 'B':0.3, 'C': 0.2}
    class_probabilities = {}

    # Keep all match counts. This will be used for prediction.
    # E.g.:{'f1=1 | A':x, 'f1=1 | A': y}
    all_counts = {}

    # Train size, used for calculating count of class
    train_size = 0

    # Equivalent sample size
    e = 1

    def __init__(self, e: int = 1):
        self.e = e

    def _calc_class_probabilities(self, classes: pd.Series) -> None:
        """Calculate probability of occurrency for each class.

        Parameters
        ----------
        classes: pd.Series
            pandas series contaning the categorized classes
        """
        labels = classes.value_counts()
        n = classes.shape[0]
        self.class_probabilities = {label: count/n
                                    for label, count in labels.items()}

    def train(self, data: pd.DataFrame, labels: str):
        """
        Parameters
        ----------
        data: pandas dataframe
            dataframe containg all train data
        labels: str
            column name
        """
        self.train_size = data.shape[0]
        # Extract classes
        classes = data.loc[:, labels]
        # Count classes probabilities
        self._calc_class_probabilities(classes)
        # For each class, calculate  match count for any column
        # e.g.: P(x=1 | C)
        for c in self.class_probabilities.keys():
            # Filter entries classified as c
            new_df = data.query('classificacao_acidente == {}'.format(c))
            # Remove target columns
            new_df.drop(labels=[labels], axis=1)
            class_counts = {}
            # For each column, calulate matches
            class_counts = {'{}={}|{}'.format(column, val, c):
                            new_df[column][new_df[column] == val].shape[0]
                            for column in new_df.columns
                            for val in new_df[column].unique()
                            }

            self.all_counts.update(class_counts)

    def _predict(self, query: pd.Series) -> str:
        """ Perform classification of one entry

        Parameters
        ----------
        query: pd.Series
            entry used for prediction

        Return
        ------
        label: str
            label for class classified for query
        """

        # Init values
        label = ""
        max_prob = 0.0

        # For each class A, calculate probability using formula
        # P(A) * P(feature_1=val1 | A) ... * (P(feature_2=val2 | A)
        for c in self.class_probabilities.keys():
            # Recover P(A)
            prob_class = self.class_probabilities[c]
            # Calculate P(feature=val | A) for each feature
            # using m-estimate
            for feature, val in query.items():
                # Count number of examples that from class c
                # that have value 'val' for feature 'feature'
                match = '{}={}|{}'.format(feature, val, c)
                n_match = self.all_counts.get(match, 0)

                p_class = self.class_probabilities[c]

                # Count number of examples of class c
                n_class = p_class * self.train_size

                # Estimate probabilty
                prob = (n_match + self.e * p_class) / (n_class + self.e)

                # Calculate probability of example being classified as c
                prob_class *= prob

            assert(1 > prob_class >= 0)
            # Keep the maximum probability
            if(prob_class >= max_prob):
                label = c
                max_prob = prob_class

        return label

    def test(self, query: pd.DataFrame, actual_labels: list) -> list:
        """Use the model for prediction.

        Parameters
        ----------
        query: pd.DataFrame
            features
        actual_labels: list
            true labels for query

        Return
        ------
        labels: list
            list of label for class classified for each entry
        """
        labels = []
        n = query.shape[0]
        i = 1
        for index, row in query.iterrows():
            label = self._predict(row)
            labels.append(label)
            print('Testing {}/{}'.format(i, n))
            i += 1

        self._confusion_matrix(actual_labels.values, labels, self.class_probabilities.keys())
        return labels


# class SVM(Model):
#     """ Implementation of support vector machine
#     algorithm for classification.

#     """

#     # regularization parameter that trades off margin size and training error
#     C = 1

#     # error
#     e = 0.01

#     def __init__(self, C: int = 0, e: int = 0.01):
#         self.C = C
#         self.e = e

#     def _loss_function(self, yn: int, y: int) -> int:
#         """ Loss function that returns 0 if yn equals y, and 1 otherwise.

#         Parameters
#         ----------
#         yn: int
#             classified label.
#         y: int
#             actual label.

#         Return
#         ------
#         loss: int
#             0 if yn equals y, and 1 otherwise.
#         """

#         return (0 if yn == y else 100)

#     def _separating_oracle(self, x_i, w, y_i, Y):
#         return y_i

#     def _argmin(self, w, slack, i, W):
#         return w, slack

#     def _psi(self, x: list, y: int) -> list:
#         """ Parameter vector that stacks x into position y.
#         Parameters
#         ----------
#         x: list
#             list of features
#         y: int
#             label
#         Return
#         ------
#         vector
#         """

#         vec = np.zeros((self.n_labels, self.n_features))
#         vec[y] = x
#         return vec

#     def _kernel(self, a: np.array, b:):
#         """ Linear kernel. """
#         return np.inner(a, b)

#     def _classify():
#         dist=0
#         for(i=1;i<model->sv_num;i+=1) {  
#             dist+=self._kernel(&model->kernel_parm,model->supvec[i],ex)*model->alpha[i];
#         }
#         return(dist - self.b);

#     def _classification_score(self, x,y,sm,sparm):
#         """Return an example, label pair discriminant score."""
#         score = self._classify(psi(x,y,sm,sparm))
#         global thecount
#         thecount += 1
#         if (sum(abs(w) for w in sm.w)):
#             import pdb; pdb.set_trace()
#         return score

#     def _predict(self, x):
#         """Given a pattern x, return the predicted label."""
#         scores = [(classification_score(x,c,sm,sparm), c)
#                 for c in xrange(1,self.num_labels+1)]
#         # Return the label with the max discriminant value.
#         return max(scores)[1]

#     def train(self, data: pd.DataFrame, labels: list):
#         """
#         Parameters
#         ----------
#         data: pandas dataframe
#             dataframe containg all train data
#         labels: list
#             list of classes columns indexes
#         """
#         n = data.shape[0]  # get sample size
#         self.n_features = n - 1

#         # Get x and y
#         x = data.drop(labels=data.columns[labels], axis=1).values
#         y = data.iloc[:, labels]

#         # Recover set of labels
#         Y = data.classificacao_acidente.unique()
#         self.n_labels = len(Y)

#         # Init w
#         self.w = np.zeros(self.n_features)

#         # Init params
#         W = [{} for i in range(n)]  # init set of constraints
#         slacks = [0 for i in range(n)]  # init slack variables
#         self.slack = 0  # stub

#         # iterate until no W has changed during iteration
#         repeat = True
#         while repeat:
#             repeat = False
#             for i in range(n):
#                 # Find the most violated constraint
#                 y_hat = self._separating_oracle(x[i], self.w, y.iloc[:, i], Y)
#                 # If this constraint is violated by more than the
#                 # desired precision, the constraint is added to the working set
#                 var1 = [self._psi(x[i], y[i]) - self._psi(x[i], y_hat)]
#                 var = 1 - np.dot(self.w, var1)
#                 precision = slacks[i] + self.e
#                 if self._loss_function(y[i], y_hat) * var > precision:
#                     repeat = True
#                     W[i].append(y_hat)
#                     self.w, self.slack = self._argmin(self.w, self.slack, i, W)


class KNN(Model):

    def __init__(self, metric: str, k: int):
        self.metric = metric
        self.k = k

    def _euclidean_distance(self, point1, point2):
        dist = 0.0
        for i in range(len(point1)):
            dist += (point2[i] - point1[i]) * (point2[i] - point1[i])
        return np.sqrt(dist)

    def _calculate_distance(self, point1, point2):
        if self.metric == 'euclidean':
            return self._euclidean_distance(point1, point2)

    def train(self, data: pd.DataFrame, labels: str):
        """
        Parameters
        ----------
        data: pandas dataframe
            dataframe containg all train data
        labels: str
            column name
        """
        self.train_size = data.shape[0]
        # Extract classes
        classes = data.loc[:, labels]
        # Keep the instances
        self.instances = data

    def _predict(self, query: pd.Series) -> str:
        """ Perform classification of one entry

        Parameters
        ----------
        query: pd.Series
            entry used for prediction

        Return
        ------
        label: str
            label for class classified for query
        """
        neighbors = []

        # Measure the distance from the new data to all others data 
        # that is already classified
        distances = [(self._calculate_distance(query,instance[:-2]), instance[-1]) 
                      for instance in self.instances]
        
        # Get the K smaller distances
        neighbors = distances.sort(key=lambda tup: tup[0])[:self.k]
        # Check the list of classes had the shortest distance and 
        # count the amount of each class that appears
        count_classes = { c: count}
        for neighbor in neighbors:
            c = neighbor[1] #  get class
            count_classes[c] = count_classes.get(c, 0) + 1
        # Takes as correct class the class that appeared the most times
        label = max(count_classes.items(), key=operator.itemgetter(1))[0]

        return label

    def test(self, query: pd.DataFrame, actual_labels: list) -> list:
        """Use the model for prediction.

        Parameters
        ----------
        query: pd.DataFrame
            features
        actual_labels: list
            true labels for query

        Return
        ------
        labels: list
            list of label for class classified for each entry
        """
        labels = []
        n = query.shape[0]
        i = 1
        for index, row in query.iterrows():
            label = self._predict(row)
            labels.append(label)
            print('Testing {}/{}'.format(i, n))
            i += 1

        self._confusion_matrix(actual_labels.values, labels, self.class_probabilities.keys())
        return labels