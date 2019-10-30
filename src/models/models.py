import pandas as pd
import numpy as np


class NaiveBayes:
    """ Implementation of a naïve version of the Bayesian belief network.
    The features are assumed to be independent.

    P(B | A) = P( B /\ A) / P(A)
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

    def _calc_class_probabilities(self, classes: pd.DataFrame) -> None:
        """Calculate probability of occurrency for each class.

        Parameters
        ----------
        classes: pandas dataframe
            pandas dataframe contaning the categorized classes
        """
        labels = classes.idxmax(axis=1).value_counts()
        n = classes.shape[0]
        self.class_probabilities = {label: count/n
                                    for label, count in labels.items()}

    def train(self, data: pd.DataFrame, labels: list):
        """
        Parameters
        ----------
        data: pandas dataframe
            dataframe containg all train data
        labels: list
            list of classes columns indexes
        """
        self.train_size = data.shape[0]
        # Extract classes
        classes = data.iloc[:, labels[0]].unique()
        assert(classes.shape[1] == 4)
        # Count classes probabilities
        self._calc_class_probabilities(classes)
        # For each class, calculate  match count for any column
        # e.g.: P(x=1 | C)
        for c in self.class_probabilities.keys():
            # Filter entries classified as c
            filtered_df = data.query('{} == 1'.format(c))
            # Remove target columns
            filtered_df.drop(labels=data.columns[labels], axis=1)
            class_counts = {}
            # For each column, calulate matches
            class_counts = {'{}={}|{}'.format(column, val, c):
                            filtered_df[column][filtered_df[column] == val].shape[0]
                            for column in filtered_df.columns
                            for val in filtered_df[column].unique()
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

    def _confusion_matrix(self, actual_labels: list,
                                predicted_labels: list) -> None:
        """
        """
        n_labels = len(self.class_probabilities.keys())
        # Create attribute as dict and keep labels indexes
        self.confusion_matrix = { 
            key: value for value, key in enumerate(self.class_probabilities.keys())
        }
        # Init matrix
        matrix = [
            [0 for i in range(n_labels)] for j in range(n_labels)
        ]

        # Fill matrix
        # Rows: actual class
        # Columns: predicted class
        for ac in actual_labels:
            for pc in predicted_labels:
                ac_index = self.confusion_matrix[ac]
                pc_index = self.confusion_matrix[pc]
                matrix[ac_index][pc_index] += 1

        # Update dict
        self.confusion_matrix['matrix'] = matrix
        print(matrix)


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
            print('{}/{}'.format(i, n))
            i += 1

        self._confusion_matrix(actual_labels, labels)
        print(labels)
        print(actual_labels)
        return labels


class SVM:
    """ Implementation of support vector machine
    algorithm for classification.

    """

    # regularization parameter that trades off margin size and training error
    C = 1

    # error
    e = 0.01

    def __init__(self, C: int = 0, e: int = 0.01):
        self.C = C
        self.e = e

    def _loss_function(self, yn: int, y: int) -> int:
        """ Loss function that returns 0 if yn equals y, and 1 otherwise.

        Parameters
        ----------
        yn: int
            classified label.
        y: int
            actual label.

        Return
        ------
        loss: int
            0 if yn equals y, and 1 otherwise.
        """

        return (0 if yn == y else 100)

    def _separating_oracle(self, x_i, w, y_i, Y):
        return y_i

    def _argmin(self, w, slack, i , W):
        return w, slack

    def _phi(self, x: list, y: int) -> list:
        """ Parameter vector that stacks x into position y.
        Parameters
        ----------
        x: list
            list of features
        y: int
            label
        Return
        ------
        vector
        """

        vec = np.zeros( (self.n_labels, self.n_features))
        vec[y] = x
        return vec

    def train(self, data: pd.DataFrame, labels: list):
        """
        Parameters
        ----------
        data: pandas dataframe
            dataframe containg all train data
        labels: list
            list of classes columns indexes
        """
        n = data.shape[0]  # get sample size
        self.n_features = n - 1

        # Get x and y
        x = data.drop(labels=data.columns[labels], axis=1).values
        y = data.iloc[:, labels]

        # Recover set of labels
        Y = data.classificacao_acidente.unique()
        self.n_labels = len(Y)

        # Init w
        self.w = np.zeros(self.n_features)

        # Init params
        W = [ {} for i in range(n)]  # init set of constraints
        slacks = [ 0 for i in range(n)]  # init slack variables 

        # iterate until no W has changed during iteration
        repeat = True
        while repeat:
            repeat = False
            for i in range(n):
                # Find the most violated constraint
                y_hat = self._separating_oracle(x[i], self.w, y.iloc[:,i], Y)
                # If this constraint is violated by more than the
                # desired precision, the constraint is added to the working set
                var = np.dot(self.w, [self._phi(x[i], y[i]) - self._phi(x[i] , y_hat)])
                if self._loss_function(y_i, y_hat) * (1 - var) > (slacks[i] + self.e):
                    repeat = True
                    W[i].append(y_hat)
                    self.w, slack = self._argmin(self.w, slack, i , W)
                    


