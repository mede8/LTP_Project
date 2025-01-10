import json
import nltk
import logging
from nltk.tokenize import word_tokenize
from pathlib import Path
import pandas as pd


root = Path(__file__).resolve().parent.parent
log_dir = root / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
logs_path = log_dir / 'preprocessing_info.log'

format_style = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(filename=logs_path, level=logging.INFO,
                    format=format_style)


class Logger:
    """Simple logger class."""
    @staticmethod
    def log_info(message):
        logging.info(message)

    @staticmethod
    def log_error(message):
        logging.error(message)


class Preprocessor:
    def __init__(self, data_path: str, naughty_file: str = None) -> None:
        """
        Initialize the Preprocessor class.

        :param naughty_url: URL to fetch the obscene words JSON file.
        :param naughty_file: local file path to the naughty words JSON file.
        """
        Logger.log_info("Initializing Preprocessing method.")
        nltk.download('punkt')

        self.data = self.load_data(data_path)

        if naughty_file:
            self.obscene_words = self._load_obscene_file(naughty_file)
        else:
            Logger.log_error("No obscene words file provided.")

    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load data from a pandas DataFrame.

        :param data: pandas DataFrame.
        :return: pandas DataFrame.
        """
        Logger.log_info("Loading data.")
        try:
            data = pd.read_csv(data_path)
            Logger.log_info("Data loaded successfully.")
            return data
        except Exception as e:
            Logger.log_error(f"Error while loading data: {e}")

    @staticmethod
    def _load_obscene_file(file_path: str) -> set:
        """
        Load obscene words from a local JSON file.

        :param file_path: path to the JSON file.
        :return: set of obscene words.
        """
        Logger.log_info(f"Loading obscene words from {file_path}.")
        with open(file_path, 'r') as f:
            return set(json.load(f))

    def contains_obscene_words(self, text: str) -> bool:
        """
        Check if a text contains any obscene words.

        :param text: input text.
        :return: true if text contains obscene words, false otherwise.
        """
        words = word_tokenize(text.lower())
        return any(word in self.obscene_words for word in words)

    def remove_obscene_data(self) -> pd.DataFrame:
        """
        Remove rows from a DataFrame containing obscene words.

        :param data: pandas DataFrame.
        :param column_name: Column name to check for obscene words.
        :return: Cleaned DataFrame.
        """
        # count he number of rows with obscene words
        obscene_count = (
            self.data['utterance'].apply(self.contains_obscene_words).sum()
        )
        Logger.log_info(f"Number of rows with obscene words: {obscene_count}")
        print(f"Number of rows with obscene words: {obscene_count}")

        # remove the obscene rows
        cleaned_data = self.data[
            ~self.data['utterance'].apply(self.contains_obscene_words)]

        Logger.log_info("Number of rows after removing " +
                        f"obscene words: {cleaned_data.shape[0]}")
        print("Number of rows after removing " +
              f"obscene words: {cleaned_data.shape[0]}")

        return cleaned_data


if __name__ == "__main__":
    data_path = root / 'Data' / 'Artemis.csv'
    preprocessor = Preprocessor(data_path,
                                naughty_file=root / 'Data' / 'en.json')

    cleaned_data = preprocessor.remove_obscene_data()
    # save the cleaned data to a new file
    new_path = root / 'Data' / 'Artemis-cleaned.csv'
    cleaned_data.to_csv(new_path, index=False)
