from loguru import logger
import pandas as pd
from pathlib import Path


class FileHandler:
    def __init__(self, file_path: str):
        """
        A file handler class for reading and writing CSV files using pathlib.

        :param file_path: The path to the CSV file as a string or Path object.
        """
        self.file_path = Path(file_path)

    def read_csv(self) -> pd.DataFrame:
        """
        Reads a CSV file and logs the action.

        :return: A pandas DataFrame.
        """
        try:
            df = pd.read_csv(self.file_path)
            logger.info(f"CSV file read successfully: {self.file_path}")
            return df
        except Exception as e:
            logger.error(f"Error reading the CSV file from {self.file_path}: {e}")
            raise

    def write_csv(self, df, index=False):
        """
        Writes a DataFrame to a CSV file and logs the action.

        :param df: The pandas DataFrame to write.
        :param index: Whether to write row indices.
        """
        try:
            df.to_csv(self.file_path, index=index)
            logger.info(f"CSV file written successfully: {self.file_path}")
        except Exception as e:
            logger.error(f"Error writing to the CSV file at {self.file_path}: {e}")
            raise
