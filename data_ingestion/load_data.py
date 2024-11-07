from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.utils import AnalysisException
import logging

class DataLoader:
    def __init__(self, file_path: str, columns: list = None):
        """
        Initializes the DataLoader with a file path and optional column list.
        
        Parameters:
        - file_path (str): Path to the CSV file.
        - columns (list): Optional list of columns to load; loads all if None.
        """
        self.file_path = file_path
        self.columns = columns
        self.spark = SparkSession.builder \
            .appName("DataLoader") \
            .getOrCreate()
        self.data = None
        logging.basicConfig(level=logging.INFO)

    def load_data(self) -> DataFrame:
        """
        Loads data from the specified CSV file using PySpark.
        
        Returns:
        - DataFrame: Loaded data as a PySpark DataFrame.
        """
        try:
            if self.columns:
                self.data = self.spark.read.csv(self.file_path, header=True).select(*self.columns)
            else:
                self.data = self.spark.read.csv(self.file_path, header=True)
            logging.info(f"Data loaded successfully from {self.file_path}")
            return self.data
        except AnalysisException as e:
            logging.error(f"Error loading data: {e}")
            raise

    def check_nulls(self) -> DataFrame:
        """
        Checks for null values in the DataFrame and returns rows with any nulls.

        Returns:
        - DataFrame: A PySpark DataFrame containing rows with null values.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        null_data = self.data.filter(" OR ".join([f"{col} IS NULL" for col in self.data.columns]))
        logging.info("Null value check complete.")
        return null_data

    def show_data_sample(self, num_rows: int = 5):
        """
        Displays a sample of the data.

        Parameters:
        - num_rows (int): Number of rows to display.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        self.data.show(num_rows)

    def close_session(self):
        """
        Stops the Spark session.
        """
        self.spark.stop()
        logging.info("Spark session stopped.")

