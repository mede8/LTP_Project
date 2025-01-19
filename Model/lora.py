import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from Modules.data_loader import DataLoader as dl
from Modules.story_gen_model import StoryGenerationModel as sgm

from pathlib import Path
import logging
import optuna


def objective(trial):
    hyperparams = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        "batch_size": trial.suggest_int("batch_size", 16, 64, step=16),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 10),
    }

    logger.info(f"Trial started with hyperparameters: {hyperparams}")

    dataset_split = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset_split["train"]
    val_dataset = dataset_split["test"]

    model = sgm(log_file_path="../Model/lora_training.log")
    model.fine_tune(train_dataset, val_dataset, hyperparams)

    metrics = model.evaluate_model(val_dataset)
    logger.info(f"Trial completed with metrics: {metrics}")
    return metrics["bertscore_f1"]


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    log_dir = root / 'Logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    logs_path = log_dir / 'lora_training.log'

    logging.basicConfig(
        filename=logs_path,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger()

    file_path = os.path.join(os.path.dirname(__file__), '../Data/Artemis-cleaned.csv')

    logger.info("Loading dataset...")
    dataset = dl.load_artemis_dataset(file_path)

    logger.info("Starting hyperparameter tuning...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3)

    logger.info("Training final model with best hyperparameters...")
    best_hyperparams = study.best_params
    logger.info(f"Best Hyperparameters: {best_hyperparams}")

    dataset_split = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset_split["train"]
    val_dataset = dataset_split["test"]

    model = sgm(log_file_path=logs_path)
    model.fine_tune(train_dataset, val_dataset, best_hyperparams)

    logger.info("Generating and evaluating stories...")
    for i, example in enumerate(val_dataset.select(range(3))):
        input_prompt = example["input_prompt"]
        emotion = example.get("emotion", "neutral")
        reference_story = example["target_output"]
        generated_story = model.generate_story(input_prompt, emotion)
        logger.info(f"Sample {i + 1}: Input: {input_prompt}, Reference: {reference_story}, Generated: {generated_story}")
        print(f"\nSample {i + 1}:")
        print(f"Input Prompt: {input_prompt}")
        print(f"Emotion: {emotion}")
        print(f"Reference Story: {reference_story}")
        print(f"Generated Story: {generated_story}")
