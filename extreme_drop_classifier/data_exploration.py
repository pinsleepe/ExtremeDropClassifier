from loguru import logger
from extreme_drop_classifier.file_handler import FileHandler


# TODO: maybe provide FileHandler as an argument
class DataExploration:
    def __init__(self, file_path):
        """
        Data exploration class for examining datasets from a CSV file.

        :param file_path: Path to the CSV file to explore.
        """
        file_handler = FileHandler(file_path)
        self.data = file_handler.read_csv()

    def show_info(self):
        """
        Logs the basic info of the dataframe.
        """
        logger.info("Basic DataFrame information:")
        self.data.info()

    def show_head(self, n=5):
        """
        Logs the first n rows of the dataframe.

        :param n: Number of rows to display.
        """
        logger.info(f"First {n} rows of the DataFrame:")
        head = self.data.head(n)
        logger.info(f"\n{head}")

    def show_missing_values(self):
        """
        Logs the count of missing values in the dataframe.
        """
        missing_values = self.data.isnull().sum()
        logger.info("Missing values in each column:")
        logger.info(f"\n{missing_values}")

    def show_class_distribution(self):
        """
        Logs the distribution of the 'Class' column.
        """
        class_distribution = self.data['Class'].value_counts(dropna=False)
        logger.info("Distribution of the 'Class' variable:")
        logger.info(f"\n{class_distribution}")
