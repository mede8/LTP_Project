import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from Modules.data_loader import DataLoader as dl
from Modules.flan_model import FlanT5Model as ft5

from pathlib import Path
import logging

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    log_dir = root / 'Logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    logs_path = log_dir / 'base_model_eval.log'

    logging.basicConfig(
        filename=logs_path,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger()

    file_path = os.path.join(os.path.dirname(__file__), '../Data/Artemis-cleaned.csv')
    val_dataset = dl.load_artemis_dataset(file_path)

    model = ft5()
    logger.info("Generating stories from the validation dataset...")
    print("Generating stories from the validation dataset...")

    for i, example in enumerate(val_dataset.select(range(3))):
        input_prompt = example["input_prompt"]
        reference_story = example["target_output"]
        generated_story = model.generate_story(input_prompt)
        print(f"\nSample {i + 1}:")
        print(f"Input Prompt: {input_prompt}")
        print(f"Reference Story: {reference_story}")
        print(f"Generated Story: {generated_story}")

    logger.info("Evaluating the model...")
    print("\nEvaluating the model...")
    metrics = model.evaluate_model(val_dataset)
    print("Evaluation Metrics:", metrics)
