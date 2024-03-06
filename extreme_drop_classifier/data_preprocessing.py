from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import pandas as pd
from loguru import logger

# TODO: add Handling extremely large datasets that might not fit into memory (chunk size)
# TODO: update the model that support incremental learning or online learning
# TODO: if dataset is really large use Dask, PySpark, or Ray for parallel processing.
# TODO: implement preprocessing for serving (like use the same scaler)
# TODO: add data monitoring and quality checks
# TODO: use Dagster or Airflow


class DataPreprocessing(BaseEstimator, TransformerMixin):
    """
    Enhanced data preprocessing class for time series anomaly detection.

    Responsibilities:
    - Handle missing values in 'Class' column.
    - Convert 'Class' to binary encoding (0 for normal, 1 for 'S').
    """

    def __init__(self):
        # TODO: add self.scaler = StandardScaler() to scale numerical features
        pass

    # TODO: add fit method to fit the scaler and modify the transform method to use it

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
