from loguru import logger


class FeatureEngineering:
    """
    Feature engineering class dedicated to creating new features from time series data.
    """

    def __init__(self):
        pass

    def add_time_based_features(self, X):
        """
        Adds time-based features to the DataFrame based on 'EventTime'.

        :param X: DataFrame with 'EventTime' as datetime.
        :return: DataFrame with new time-based features.
        """
        logger.info("Adding time-based features.")
        X['hour'] = X['EventTime'].dt.hour
        X['dayofweek'] = X['EventTime'].dt.dayofweek
        return X

    def add_rolling_features(self, X, window_size=5):
        """
        Adds rolling statistical features to the DataFrame.

        :param X: DataFrame with 'Measure'.
        :param window_size: Size of the rolling window.
        :return: DataFrame with rolling features.
        """
        logger.info(f"Adding rolling features with window size {window_size}.")
        X[f'rolling_mean_{window_size}'] = X['Measure'].rolling(window=window_size).mean()
        X[f'rolling_std_{window_size}'] = X['Measure'].rolling(window=window_size).std()

        # forward fill to impute the NaN values in rolling features
        X[f'rolling_mean_{window_size}'] = X[f'rolling_mean_{window_size}'].bfill().ffill()
        X[f'rolling_std_{window_size}'] = X[f'rolling_std_{window_size}'].bfill().ffill()
        return X
