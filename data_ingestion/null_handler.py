from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, regexp_extract, year
import logging

class NullHandler:
    def __init__(self, data: DataFrame):
        """
        Initializes the NullHandler with a DataFrame containing missing values.

        Parameters:
        - data (DataFrame): The Spark DataFrame with potential null values.
        """
        self.data = data
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def process_dates_and_fill_missing_values(self) -> DataFrame:
        """
        Processes the 'publishedDate' column to extract the year, sets default values for missing fields,
        and fills nulls in 'publishedYear' with the median year.

        Returns:
        - DataFrame: The processed DataFrame.
        """
        # Define the year pattern to match 4-digit years
        year_pattern = r"(\d{4})"

        # Extract the year from 'publishedDate' or set to None if not present
        self.data = self.data.withColumn(
            "publishedDate",
            when(
                col("publishedDate").rlike(year_pattern),
                regexp_extract(col("publishedDate"), year_pattern, 1)  # Extract year if present
            )
        )

        # Drop unnecessary column '_c0' if it exists
        if '_c0' in self.data.columns:
            self.data = self.data.drop('_c0')

        # Fill missing values for 'Title', 'description', and 'authors' columns
        self.data = self.data.fillna({
            "Title": "No Title",
            "description": "No Description",
            "authors": "Unknown author"
        })

        # Extract 'publishedYear' from 'publishedDate' as a numeric year
        self.data = self.data.withColumn("publishedYear", year(col("publishedDate")))

        # Calculate the median year for 'publishedYear'
        median_year = self.data.approxQuantile("publishedYear", [0.5], 0.01)[0]
        
        # Fill nulls in 'publishedYear' with the calculated median year
        if median_year is not None:
            self.data = self.data.fillna({"publishedYear": str(int(median_year))})
        
        self.logger.info("Completed processing dates and filling missing values.")
        return self.data
