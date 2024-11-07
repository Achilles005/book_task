import numpy as np
import logging

class PostProcessor:
    def __init__(self, max_value):
        """
        Initializes the PostProcessor with a logger and the max_value for inverse transformation.

        Parameters:
        - max_value (float): The maximum value used in the log transformation of the target variable.
        """
        self.max_value = max_value
        self.logger = logging.getLogger(self.__class__.__name__)

    def inverse_log_transform(self, predictions):
        """
        Applies an inverse log transformation to the predictions to revert them to their original scale.

        Parameters:
        - predictions (np.array or list): The log-transformed predictions to inverse-transform.

        Returns:
        - np.array: Transformed predictions on the original scale.
        """
        try:
            # Ensure predictions are in an array format for processing
            predictions = np.array(predictions)
            # Inverse the log transformation to return to the original scale
            transformed_predictions = self.max_value - np.exp(predictions)
            self.logger.info("Inverse log transformation applied to predictions.")
            return transformed_predictions
        except Exception as e:
            self.logger.error(f"Error applying inverse log transform: {e}")
            raise

    def post_process(self, predictions, transformations):
        """
        Applies a series of transformations to the predictions in sequence.

        Parameters:
        - predictions (np.array or list): Initial predictions.
        - transformations (list of dict): List of transformations to apply, each specified as a dict.

        Example of transformations:
        - [{"type": "inverse_log"}]

        Returns:
        - np.array: Final post-processed predictions.
        """
        for transform in transformations:
            if transform["type"] == "inverse_log":
                predictions = self.inverse_log_transform(predictions)
            else:
                self.logger.warning(f"Unknown transformation type: {transform['type']}")
        
        self.logger.info("All transformations applied successfully.")
        return predictions
