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

    def train(data: list, labels: list, e: int=4) -> float:
        """
        """
        # Extract classes
        # Calculate classes probabilities
        # For each class
        #   Calculate  
        n = 0  # number of examples that match example
        c = 0  # number of examples of classification
        p = 1  # smoothing prior
        return (n + e * p) / (c + e)
