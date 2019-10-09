import pandas as pd

class NaiveBayes:
    """ Implementation of a naïve version of the Bayesian belief network.
    The features are assumed to be independent.

    P(B | A) = P( B /\ A) / P(A)
    """

    # Keep the probabilities of each class as a dictionary.
    # E.g.: {'A': 0.5, 'B':0.3, 'C': 0.2}
    class_probabilities = {}

    # Keep all probablities. This will be used for prediction.
    # E.g.:{'f1=1 | A':x, 'f1=1 | A': y}
    all_probabilities = {}

    def _calc_class_probabilities(self, classes: pd.DataFrame) -> None:
        """Calculate probabilities for each class to occurr.

        Parameters
        ----------
        classes: pandas dataframe
            pandas dataframe contaning the categorized classes
        """
        labels = classes.idxmax(axis=1).value_counts()
        n = classes.shape[0]
        self.class_probabilities = {label: count/n
                                    for label, count in labels.items()}

    def train(self, data: pd.DataFrame, labels: list, e: int=1):
        """
        Parameters
        ----------
        data: pandas dataframe
            dataframe containg all train data
        labels: list
            list of classes columns indexes
        e: int
            equivalent sample size, used for adapting to very rare classes.
            Default value is 1.
        """
        # Extract classes
        classes = data.iloc[:, labels]
        # Calculate classes probabilities
        self._calc_class_probabilities(classes)
        print(self.class_probabilities)
        # For each class
        #   Calculate  
        n = 0  # number of examples that match example
        c = 0  # number of examples of classification
        p = 1  # smoothing prior
        return (n + e * p) / (c + e)
