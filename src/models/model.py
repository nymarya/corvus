import numpy as np

class Model():
    """Class that every model enhiterates. """

    def _confusion_matrix(self, actual_labels: list,
                        predicted_labels: list, classes:list) -> None:
        """ Creates confusion matrix for model.

        Parameters
        ----------
        actual_labels: list
            list of integers containing the true labels
        predicted_labels: list
            list of integers containing the predicted labels
        classes: list
            list of integers containing the unique labels
        """
        n_labels = len(classes)
        # Create attribute as dict and keep labels indexes
        self.confusion_matrix = {
            key: value for value, key in enumerate(classes)
        }
        # Init matrix
        matrix = [
            [0 for i in range(n_labels)] for j in range(n_labels)
        ]

        # Fill matrix
        # Rows: actual class
        # Columns: predicted class
        n_samples = len(actual_labels)
        for i in range(n_samples):
            # for sample i, update the matrix
            ac = actual_labels[i]
            pc = predicted_labels[i]
            ac_index = self.confusion_matrix[ac]
            pc_index = self.confusion_matrix[pc]
            matrix[ac_index][pc_index] += 1

        # Update dict
        self.confusion_matrix['matrix'] = matrix

        # Test
        x = np.matrix(matrix)
        x.sum()
        assert(x.sum() == len(actual_labels))

    def _accuracy(self) -> float:
        """Calculate the accuracy for the model. 

        Return
        ------
        a float representing the models' accuracy
        """
        matrix = self.confusion_matrix['matrix']
        return np.diag(matrix).sum() / np.sum(matrix)

    def _precision(self) -> list:
        """ Calculate the precision for the model.
        Return
        ------
        list of floats containing the precision for each class
        """

        result = []
        matrix = self.confusion_matrix['matrix']
        for i in range(len(matrix)):
            precision_sum = 0
            for j in range(len(matrix)):
                precision_sum += matrix[j][i]
            # Save precision
            result.append( matrix[i][i] / precision_sum)
        
        return result

    def to_string(self) -> str:
        """ Resumes all parameters of the model. """
        return ""

    def report(self) -> str:
        labels = ['com_vitimas_fatais','com_vitimas_feridas','ignorado','sem_vitimas']
        acc = self._accuracy() * 100
        s = ""
        text = f"\nAccuracy:{acc:>20.2f}%"
        for i, precision in enumerate(self._precision()):
            text += f"\nPrecision `{labels[i]}`:{precision*100:>20.2f}%"

        return text