# tests/test_post_processing.py
import pytest
from post_processing import PostProcessor

def test_inverse_log_transform():
    max_value = 100
    post_processor = PostProcessor(max_value)
    predictions = [2.3, 1.8, 2.1]  # Example log-transformed predictions
    transformed_predictions = post_processor.inverse_log_transform(predictions)
    assert transformed_predictions is not None
    assert all(p >= 0 for p in transformed_predictions)  # Ensure predictions are positive

# More Tests can be aadded