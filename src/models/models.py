import pandas as pd

class NaiveBayes:
    """ Implementation of a naïve version of the Bayesian belief network.
    The features are assumed to be independent.

    P(B | A) = P( B /\ A) / P(A)
    """

    # Keep the probabilities of each class as a dictionary.
    # E.g.: {'A': 0.5, 'B':0.3, 'C': 0.2}
    class_probabilities = {}

    # Keep all mach counts. This will be used for prediction.
    # E.g.:{'f1=1 | A':x, 'f1=1 | A': y}
    all_counts = {}

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
        # Count classes probabilities
        self._calc_class_probabilities(classes)
        print(self.class_probabilities)
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
