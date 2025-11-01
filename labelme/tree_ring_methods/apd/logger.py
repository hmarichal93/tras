# module for logging messages to a file.
import logging

class Logger:
    """Logs messages to a file. Logging module is used by default."""
    def __init__(self, filename):
        self.filename = filename
        #add timestamp to log file
        logging.basicConfig(filename=self.filename, level=logging.INFO, format='%(asctime)s %(message)s')


    def log(self, message):
        """Logs a message to a file."""
        logging.info(message)
