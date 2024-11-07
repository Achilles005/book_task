from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, HashingTF, IDF, FeatureHasher
from sparknlp.annotator import DocumentAssembler, Tokenizer, Normalizer, ViveknSentimentModel, Finisher, BertSentenceEmbeddings
from pyspark.ml.linalg import Vectors, VectorUDT
import logging

class FeatureEngineer:
    def __init__(self, data: DataFrame):
        """
        Initializes the FeatureEngineer with a DataFrame.

        Parameters:
        - data (DataFrame): The Spark DataFrame on which to perform feature engineering.
        """
        self.data = data
        self.current_year = 2024
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def create_date_features(self) -> DataFrame:
        """
        Creates date-related features: book age, published decade, published century, and published era.

        Returns:
        - DataFrame: The DataFrame with the new date-related features added.
        """
        self.data = self.data.withColumn("book_age", F.lit(self.current_year) - F.col("publishedYear"))
        self.data = self.data.withColumn("published_decade", (F.col("publishedYear") / 10).cast("int") * 10)
        self.data = self.data.withColumn("published_century", (F.col("publishedYear") / 100).cast("int") + 1)
        self.data = self.data.withColumn(
            "published_era",
            F.when(F.col("publishedYear") >= 2000, "Modern")
             .when((F.col("publishedYear") >= 1900) & (F.col("publishedYear") < 2000), "Contemporary")
             .when((F.col("publishedYear") >= 1800) & (F.col("publishedYear") < 1900), "Classic")
             .otherwise("Ancient")
        )
        self.logger.info("Date-related features created.")
        return self.data

    def encode_era_feature(self) -> DataFrame:
        """
        Encodes the 'published_era' feature using StringIndexer and OneHotEncoder.

        Returns:
        - DataFrame: The DataFrame with 'published_era' encoded.
        """
        indexer = StringIndexer(inputCol="published_era", outputCol="published_era_index")
        encoder = OneHotEncoder(inputCol="published_era_index", outputCol="published_era_encoded")
        pipeline = Pipeline(stages=[indexer, encoder])
        self.data = pipeline.fit(self.data).transform(self.data)
        self.logger.info("Encoded 'published_era' feature.")
        return self.data

    def compute_tfidf(self, column: str, output_col: str, num_features: int = 20) -> DataFrame:
        """
        Computes TF-IDF features for a specified text column.

        Parameters:
        - column (str): Column to compute TF-IDF for (e.g., 'Title' or 'description').
        - output_col (str): Output column for the TF-IDF vector.
        - num_features (int): Number of features for HashingTF.

        Returns:
        - DataFrame: DataFrame with TF-IDF features added.
        """
        hashingTF = HashingTF(inputCol=column, outputCol=f"{column}_tf", numFeatures=num_features)
        idf = IDF(inputCol=f"{column}_tf", outputCol=output_col)
        self.data = hashingTF.transform(self.data).cache()
        self.data = idf.fit(self.data).transform(self.data)
        self.logger.info(f"Computed TF-IDF for '{column}'.")
        return self.data

    def add_text_features(self) -> DataFrame:
        """
        Adds TF-IDF features for 'Title' and 'description'.

        Returns:
        - DataFrame: DataFrame with TF-IDF features added.
        """
        self.data = self.compute_tfidf(column="Title", output_col="title_tfidf_dense")
        self.data = self.compute_tfidf(column="description", output_col="desc_tfidf_dense")
        return self.data

    def add_sentiment_analysis(self) -> DataFrame:
        """
        Adds sentiment analysis features for 'Title' and 'description'.

        Returns:
        - DataFrame: DataFrame with sentiment features added.
        """
        document_title = DocumentAssembler().setInputCol("Title").setOutputCol("title_document")
        document_desc = DocumentAssembler().setInputCol("description").setOutputCol("desc_document")
        token_title = Tokenizer().setInputCols(["title_document"]).setOutputCol("title_token")
        token_desc = Tokenizer().setInputCols(["desc_document"]).setOutputCol("desc_token")
        normalizer_title = Normalizer().setInputCols(["title_token"]).setOutputCol("title_normal")
        normalizer_desc = Normalizer().setInputCols(["desc_token"]).setOutputCol("desc_normal")
        vivekn_title = ViveknSentimentModel.pretrained().setInputCols(["title_document", "title_normal"]).setOutputCol("t_sentiment")
        vivekn_desc = ViveknSentimentModel.pretrained().setInputCols(["desc_document", "desc_normal"]).setOutputCol("d_sentiment")
        finisher_title = Finisher().setInputCols(["t_sentiment"]).setOutputCols(["title_sentiment"])
        finisher_desc = Finisher().setInputCols(["d_sentiment"]).setOutputCols(["description_sentiment"])
        
        pipeline = Pipeline(stages=[
            document_title, token_title, normalizer_title, vivekn_title, finisher_title,
            document_desc, token_desc, normalizer_desc, vivekn_desc, finisher_desc
        ])
        self.data = pipeline.fit(self.data).transform(self.data)
        self.data = self.data.withColumn("title_sentiment_encoded", F.when(F.col("title_sentiment")[0] == "positive", 1).when(F.col("title_sentiment")[0] == "negative", 2).otherwise(0))
        self.data = self.data.withColumn("description_sentiment_encoded", F.when(F.col("description_sentiment")[0] == "positive", 1).when(F.col("description_sentiment")[0] == "negative", 2).otherwise(0))
        self.data = self.data.drop("title_sentiment", "description_sentiment")
        self.logger.info("Sentiment analysis completed.")
        return self.data

    def add_author_publisher_frequency(self) -> DataFrame:
        """
        Adds frequency encoding for 'authors' and 'publisher'.

        Returns:
        - DataFrame: DataFrame with frequency-encoded features.
        """
        author_publisher_counts = self.data.groupBy("authors", "publisher").agg(
            F.count("authors").alias("author_frequency"),
            F.count("publisher").alias("publisher_frequency")
        )
        self.data = self.data.join(author_publisher_counts, on=["authors", "publisher"], how="left")
        self.logger.info("Added frequency encoding for 'authors' and 'publisher'.")
        return self.data

    def add_author_publisher_embeddings(self) -> DataFrame:
        """
        Adds BERT embeddings for 'authors' and 'publisher' and combines them.

        Returns:
        - DataFrame: DataFrame with combined BERT embeddings.
        """
        document_assembler_authors = DocumentAssembler().setInputCol("authors").setOutputCol("authors_document")
        document_assembler_publishers = DocumentAssembler().setInputCol("publisher").setOutputCol("publishers_document").setCleanupMode("shrink")
        bert_embeddings_authors = BertSentenceEmbeddings.pretrained("sent_small_bert_L10_128", "en").setInputCols(["authors_document"]).setOutputCol("authors_embedding")
        bert_embeddings_publishers = BertSentenceEmbeddings.pretrained("sent_small_bert_L10_128", "en").setInputCols(["publishers_document"]).setOutputCol("publishers_embedding")
        embedding_pipeline = Pipeline(stages=[document_assembler_authors, document_assembler_publishers, bert_embeddings_authors, bert_embeddings_publishers])
        self.data = embedding_pipeline.fit(self.data).transform(self.data)
        self.data = self.data.withColumn("author_publisher_combined_embedding", F.concat(F.col("authors_embedding.embeddings"), F.col("publishers_embedding.embeddings")))
        self.logger.info("Added BERT embeddings for 'authors' and 'publisher'.")
        return self.data

    def add_category_feature_hashing(self, num_features: int = 20) -> DataFrame:
        """
        Adds hashed feature representation for 'categories'.

        Parameters:
        - num_features (int): Number of features for the FeatureHasher.

        Returns:
        - DataFrame: DataFrame with hashed features for 'categories'.
        """
        hasher = FeatureHasher(inputCols=["categories"], outputCol="Category_Index", numFeatures=num_features)
        self.data = hasher.transform(self.data)
        self.logger.info(f"Added hashed feature for 'categories'.")
        return self.data

    def drop_unnecessary_columns(self, columns_to_drop: list) -> DataFrame:
        """
        Drops specified columns from the DataFrame.

        Parameters:
        - columns_to_drop (list): List of column names to drop.

        Returns:
        - DataFrame: DataFrame with specified columns dropped.
        """
        self.data = self.data.drop(*columns_to_drop)
        self.logger.info(f"Dropped columns: {columns_to_drop}")
        return self.data

    def convert_array_to_dense_vector(self) -> DataFrame:
        """
        Converts an array column to a dense vector for 'author_publisher_combined_embedding'.

        Returns:
        - DataFrame: DataFrame with 'author_publisher_combined_embedding' as a dense vector.
        """
        def array_to_dense_vector(array):
            return Vectors.dense(array) if array else Vectors.dense([])

        array_to_vector_udf = F.udf(array_to_dense_vector, VectorUDT())
        self.data = self.data.withColumn(
            "author_publisher_combined_embedding",
            array_to_vector_udf(F.flatten("author_publisher_combined_embedding"))
        )
        self.logger.info("Converted 'author_publisher_combined_embedding' to dense vector.")
        return self.data

    def convert_sparse_to_dense_tfidf(self) -> DataFrame:
        """
        Converts SparseVector TF-IDF columns ('title_tfidf' and 'desc_tfidf') to DenseVector.

        Returns:
        - DataFrame: DataFrame with dense TF-IDF vectors for 'title_tfidf_dense' and 'desc_tfidf_dense'.
        """
        dense_vector_udf = F.udf(lambda v: Vectors.dense(v.toArray()) if v is not None else None, VectorUDT())
        self.data = self.data.withColumn("title_tfidf_dense", dense_vector_udf(F.col("title_tfidf")))
        self.data = self.data.withColumn("desc_tfidf_dense", dense_vector_udf(F.col("desc_tfidf")))
        self.data = self.data.drop("title_tfidf", "desc_tfidf")
        self.logger.info("Converted SparseVector TF-IDF columns to DenseVector.")
        return self.data

    def assemble_features(self) -> DataFrame:
        """
        Assembles specified feature columns into a single vector column for training/prediction.

        Returns:
        - DataFrame: DataFrame with all features assembled into 'assembled_features' column.
        """
        # Define final feature columns
        feature_columns = [
            "publishedYear",
            "book_age",
            "published_decade",
            "published_century",
            "author_frequency",
            "publisher_frequency",
            "published_era_encoded",
            "Category_Index",
            "title_sentiment_encoded",
            "description_sentiment_encoded",
            "title_tfidf_dense",
            "desc_tfidf_dense",
            "author_publisher_combined_embedding"
        ]

        # Filter available feature columns based on actual DataFrame columns
        available_feature_columns = [col for col in feature_columns if col in self.data.columns]
        assembler = VectorAssembler(inputCols=available_feature_columns, outputCol="assembled_features")
        self.data = assembler.transform(self.data)
        self.logger.info("Assembled final feature vector for training/prediction.")
        return self.data

    def drop_unnecessary_columns(self, columns_to_drop: list) -> DataFrame:
        """
        Drops specified columns from the DataFrame.

        Parameters:
        - columns_to_drop (list): List of column names to drop.

        Returns:
        - DataFrame: The DataFrame with specified columns dropped.
        """
        self.data = self.data.drop(*columns_to_drop)
        self.logger.info(f"Dropped columns: {columns_to_drop}")
        return self.data

    def final_processed_data(self) -> DataFrame:
        """
        Prepares the DataFrame for training or prediction by converting arrays to vectors,
        assembling features, and cleaning up unnecessary columns.

        Returns:
        - DataFrame: The prepared DataFrame ready for model training/prediction.
        """
        # Convert array columns to dense vectors
        self.convert_array_to_dense_vector()

        # Convert SparseVectors in TF-IDF columns to DenseVectors
        self.convert_sparse_to_dense_tfidf()

        # Drop any intermediate columns and redundant data
        columns_to_drop = ["authors", "publisher", "published_era", "categories"]
        self.drop_unnecessary_columns(columns_to_drop)

        # Assemble all features into a single vector
        self.assemble_features()

        return self.data
    
    def build_pipeline(self) -> Pipeline:
        """
        Builds the entire feature engineering pipeline by sequentially applying all transformations.

        Returns:
        - Pipeline: A Spark ML Pipeline object containing all feature engineering stages.
        """
        self.logger.info("Building feature engineering pipeline.")
        
        # Define each step as a stage
        stages = [
            self.create_date_features(),
            self.encode_era_feature(),
            self.add_text_features(),
            self.add_sentiment_analysis(),
            self.add_author_publisher_frequency(),
            self.add_author_publisher_embeddings(),
            self.add_category_feature_hashing(),
            self.convert_array_to_dense_vector(),
            self.convert_sparse_to_dense_tfidf(),
            self.assemble_features(),
            self.drop_unnecessary_columns(["authors", "publisher", "published_era", "categories"])
        ]
        
        # Build and return the pipeline
        pipeline = Pipeline(stages=stages)
        self.logger.info("Feature engineering pipeline built successfully.")
        return pipeline