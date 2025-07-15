import pytest
from ml.model import train_model
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from ml.model import compute_model_metrics
from sklearn.model_selection import train_test_split
import pandas as pd

def test_model():
    """
    # Test that a RandomForestClassifier is returned from the train_model function.
    """

    # Create dummy data
    X_train = np.random.rand(100, 10)  # 100 samples, 10 features
    y_train = np.random.randint(0, 2, size=100)  # Binary labels

    # Train the model
    model = train_model(X_train, y_train)

    # Check if the model is an instance of RandomForestClassifier
    assert isinstance(model, RandomForestClassifier)


def test_statistics():
    """
    Test that the compute_model_metrics function correctly computes precision, recall, and fbeta.
    """
    # Create dummy data
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])

    # Compute metrics
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    # Manual calculation verification:
    # TP=2 (positions 1,4), FP=0, FN=1 (position 2), TN=2 (positions 0,3)
    # Precision = TP/(TP+FP) = 2/2 = 1.0
    # Recall = TP/(TP+FN) = 2/3 = 0.6667
    # F1 = 2*(precision*recall)/(precision+recall) = 2*(1.0*0.6667)/(1.0+0.6667) = 0.8
    
    assert precision == pytest.approx(1.0, abs=1e-10)
    assert recall == pytest.approx(0.6666666666666666, abs=1e-10)
    assert fbeta == pytest.approx(0.8, abs=1e-10)


def test_split():
    """
    Test the data is split correctly into training and testing sets.
    """
    # Create dummy data
    data = pd.DataFrame({
        'feature1': range(100),
        'feature2': range(100, 200),
        'label': [0, 1] * 50
    })

    # Split the data
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    # Check the sizes of the splits
    assert len(train) == 80
    assert len(test) == 20

    # Check that both sets contain both labels (more realistic test)
    assert set(train['label'].unique()) == {0, 1}
    assert set(test['label'].unique()) == {0, 1}
    
    # Check that the data doesn't overlap
    assert len(set(train.index).intersection(set(test.index))) == 0
    
    # Check that all original data is preserved
    assert len(set(train.index).union(set(test.index))) == 100
