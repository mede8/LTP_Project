import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from Modules.flan_model import FlanT5Model
from Modules.data_loader import DataLoader as dl

from evaluate import load
from pathlib import Path
import logging
import pandas as pd


root = Path(__file__).resolve().parent.parent
log_dir = root / 'Logs'
log_dir.mkdir(parents=True, exist_ok=True)
logs_path = log_dir / 'cot_wrapper_info.log'

logging.basicConfig(
    filename=logs_path,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger()


COT_PROMPT = (
    "Input: {input_description}\n"
    "Reasoning steps:\n"
    "1. Identify key elements and describe them (e.g. objects, environment).\n"
    "2. Analyze how these elements relate and interact with each other.\n"
    "3. Develop a logical sequence of events based on these interations.\n"
    "4. Write a meaningful and coherent story that ties the story together.\n"
    "Output: Generate a vivid and coherent story based on the above steps."
)


class CoTWrapper:
    def __init__(self, baseline_model: FlanT5Model) -> None:
        logger.info("Initializing CoT prompting.")
        self.baseline_model = baseline_model

    def generate_with_cot(self, input_caption: str) -> str:
        """
        Generate a story using the CoT prompting.

        :param input_caption: the image caption.
        :return: generated story.
        """
        prompt = COT_PROMPT.format(input_description=input_caption)
        return self.baseline_model.generate_story(prompt)

    def evaluate_with_cot(self, val_dataset: pd.DataFrame) -> dict:
        """
        Evaluate the model using the CoT prompting with
        ROUGE and BLEU metrics.

        :param dataset: the validation dataset.
        :return: evaluation metrics.
        """
        logger.info("Evaluating the model using CoT prompting.")
        rouge = load("rouge")
        bleu = load("bleu")
        predictions, references = [], []

        for example in val_dataset:
            input_caption = example["input_caption"]
            reference_story = example["target_output"]
            generated_story = self.generate_with_cot(input_caption)

            predictions.append(generated_story)
            references.append(reference_story)

        try:
            rouge_score = rouge.compute(predictions=predictions,
                                        references=references)
            bleu_score = bleu.compute(predictions=predictions,
                                      references=references)
            logger.info("Evaluation completed with ROUGE score: " +
                            f"{rouge_score} and BLEU score: {bleu_score}.")
        except Exception as e:
            logger.error(f"Error while evaluating the model: {e}")
            rouge_score = None
            bleu_score = None

        return {"ROUGE Score": rouge_score, "BLEU Score": bleu_score}


if __name__ == "__main__":
    """
    Main function to test the CoTWrapper class.
    Loads the dataset, augements it with CoT prompting,
    trains the model with CoT prompting, and evaluates.
    """
    file_path = os.path.join(
        os.path.dirname(__file__), '../Data/Artemis.csv'
    )

    def augment_with_cot(example):
        example["input_prompt"] = COT_PROMPT.format(
            input_description=example['input_prompt'])
        return example

    val_dataset = dl.load_artemis_dataset(file_path)
    val_dataset = val_dataset.map(augment_with_cot)

    baseline_model = FlanT5Model(model_name="google/flan-t5-large")
    cot_wrapper = CoTWrapper(baseline_model)

    logger.info("Training the model with CoT prompting.")
    train_df = dl.load_artemis_dataset(file_path).map(augment_with_cot)
    cot_wrapper.baseline_model.train(train_df, epochs=5, batch_size=16)

    logger.info("Evaluating the model with CoT prompting.")
    cot_metrics = cot_wrapper.evaluate_with_cot(val_dataset)
    print(cot_metrics)
