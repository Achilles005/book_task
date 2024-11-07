import logging
import sys

class LoggerConfig:
    def __init__(self, log_level=logging.INFO, log_to_file=False, log_filename="app.log"):
        """
        Initializes the LoggerConfig with specified settings and configures logging.

        Parameters:
        - log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        - log_to_file (bool): Whether to log messages to a file.
        - log_filename (str): Filename for the log file, if log_to_file is True.
        """
        self.log_level = log_level
        self.log_to_file = log_to_file
        self.log_filename = log_filename
        self.configure_logging()

    def configure_logging(self):
        """
        Configures the root logger with the specified settings.
        """
        # Define a basic logging format
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(log_format)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)

        # Remove any existing handlers (to avoid duplicate logs)
        if root_logger.hasHandlers():
            root_logger.handlers.clear()

        # StreamHandler for console output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # FileHandler for file output, if needed
        if self.log_to_file:
            file_handler = logging.FileHandler(self.log_filename)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        logging.info("Logging setup is completed.")

    def get_logger(self, name):
        """
        Returns a logger with the specified name, using the configured settings.

        Parameters:
        - name (str): Name of the logger.

        Returns:
        - logging.Logger: Configured logger instance.
        """
        return logging.getLogger(name)
