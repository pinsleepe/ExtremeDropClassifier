from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import pandas as pd
from loguru import logger


class DataPreprocessing(BaseEstimator, TransformerMixin):
    """
    Enhanced data preprocessing class for time series anomaly detection.

    Responsibilities:
    - Handle missing values in 'Class' column.
    - Convert 'Class' to binary encoding (0 for normal, 1 for 'S').
    """

    def __init__(self):
        pass

    def transform(self, X):
        logger.info("Starting data transformation.")
        # Make a copy to avoid changes to original data
        X = X.copy()

        # Handle missing values in 'Class'
        X['Class'] = X['Class'].fillna('N')  # Modified to avoid inplace=True
        logger.info("Missing values in 'Class' column handled.")

        # Convert 'Class' to binary encoding
        X['Class'] = X['Class'].apply(lambda x: 1 if x == 'S' else 0)
        logger.info("'Class' column converted to binary encoding.")

        # Convert 'EventTime' to datetime
        X['EventTime'] = pd.to_datetime(X['EventTime'])
        logger.info("'EventTime' column converted to datetime.")

        logger.info("Data transformation completed.")
        return X
