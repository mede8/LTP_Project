import pandas as pd
from datasets import Dataset


class DataLoader:
    @staticmethod
    def load_artemis_dataset(file_path):
        """
        Load the Artemis dataset from a CSV file
        and prepare it for training/testing.
        """
        df = pd.read_csv(file_path)
        df["input_prompt"] = (
            df["emotion"] + ", " +
            df["art_style"] + ": " +
            df["utterance"]
        )
        df["target_output"] = df["utterance"]
        dataset = Dataset.from_pandas(df)
        dataset = dataset.train_test_split(test_size=0.1)
        return dataset["test"]
