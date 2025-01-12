import sys
import os
from pathlib import Path
import torch
from peft import get_peft_model, LoraConfig, TaskType
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from sklearn.model_selection import train_test_split
import optuna
from datasets import Dataset
import matplotlib.pyplot as plt
from evaluate import load
from sentence_transformers import SentenceTransformer, util

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from Modules.data_loader import DataLoader as dl
import logging

log_file_path = "/scratch/s5107318/LTP/LTP_Project/Model/lora_training.log"
logging.basicConfig(
    filename=log_file_path,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger()

os.environ["WANDB_DISABLED"] = "true"

def create_lora_model(base_model):
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=32,
        target_modules=["q", "k", "v", "o", "wi", "wo"],
        lora_dropout=0.1,
    )
    try:
        return get_peft_model(base_model, lora_config)
    except ValueError as e:
        logger.error(f"Error in LoRA setup: {e}")
        logger.info("Inspecting model modules...")
        for name, module in base_model.named_modules():
            logger.debug(name)
        raise

class StoryGenerationModel:
    def __init__(self, model_name="google/flan-t5-small"):
        logger.info(f"Initializing model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def preprocess_dataset(self, dataset):
        def tokenize_function(example):
            inputs = self.tokenizer(
                example["input_prompt"],
                max_length=128,
                truncation=True,
                padding="max_length",
            )
            targets = self.tokenizer(
                example["target_output"],
                max_length=128,
                truncation=True,
                padding="max_length",
            )
            inputs["labels"] = targets["input_ids"]
            return inputs

        logger.info("Preprocessing dataset...")
        return dataset.map(tokenize_function, remove_columns=dataset.column_names)

    def fine_tune(self, train_dataset, val_dataset, hyperparams):
        logger.info("Applying LoRA configuration...")
        self.model = create_lora_model(self.model)

        train_dataset = self.preprocess_dataset(train_dataset)
        val_dataset = self.preprocess_dataset(val_dataset)

        training_args = TrainingArguments(
            output_dir="/scratch/s5107318/LTP/LTP_Project/Results",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=hyperparams["learning_rate"],
            per_device_train_batch_size=hyperparams["batch_size"],
            num_train_epochs=hyperparams["num_train_epochs"],
            logging_dir="/scratch/s5107318/LTP/LTP_Project/Results/logs",
            save_total_limit=2,
            remove_unused_columns=False,
            fp16=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
        )

        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed.")

    def generate_story(self, input_prompt, emotion):
        logger.info(f"Generating story for input prompt: {input_prompt} with emotion: {emotion}")
        inputs = self.tokenizer(
            f"Write a detailed and emotional story that conveys {emotion}: {input_prompt}",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_length=150,
            min_length=120,
            num_beams=4,
            temperature=0.8,
            no_repeat_ngram_size=3,
            repetition_penalty=1.5,
        )

        story = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        story = self._ensure_paragraph_length(story, max_lines=5)
        logger.info(f"Generated story: {story}")
        return story

    def _ensure_paragraph_length(self, story, max_lines=5):
        lines = story.split('. ')
        if len(lines) > max_lines:
            story = '. '.join(lines[:max_lines]) + '.'
        elif len(lines) < max_lines:
            story += ' ' + ' '.join(['...'] * (max_lines - len(lines)))
        return story

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

    model = StoryGenerationModel()
    model.fine_tune(train_dataset, val_dataset, hyperparams)

    metrics = evaluate_model(model, val_dataset)
    logger.info(f"Trial completed with metrics: {metrics}")
    return metrics["bertscore_f1"]

def evaluate_model(model, val_dataset):
    predictions = []
    references = []
    dissimilarity_scores = []

    logger.info("Evaluating model... Limiting to 10 samples for logging.")
    for i, example in enumerate(val_dataset.select(range(10))):
        input_prompt = example["input_prompt"]
        emotion = example.get("emotion", "neutral")
        reference_story = example["target_output"]
        generated_story = model.generate_story(input_prompt, emotion)
        predictions.append(generated_story)
        references.append(reference_story)

        input_similarity = compute_similarity(input_prompt, generated_story)
        dissimilarity_scores.append(1 - input_similarity)

        logger.info(f"Sample {i + 1}: Generated story: {generated_story}, Similarity to input: {input_similarity:.2f}")

    bertscore = load("bertscore")
    scores = bertscore.compute(predictions=predictions, references=references, lang="en")

    logger.info(f"Evaluation scores: {scores}")
    plot_results(scores, dissimilarity_scores)

    return {
        "bertscore_f1": sum(scores["f1"]) / len(scores["f1"]),
        "dissimilarity": sum(dissimilarity_scores) / len(dissimilarity_scores),
    }

def compute_similarity(text1, text2):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings1 = model.encode([text1], convert_to_tensor=True)
    embeddings2 = model.encode([text2], convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    return cosine_scores.item()

def plot_results(scores, dissimilarity_scores):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.bar(range(len(scores["f1"])), scores["f1"], color="blue")
    plt.title("BERTScore F1 for Generated Stories")
    plt.xlabel("Sample Index")
    plt.ylabel("BERTScore F1")

    plt.subplot(1, 2, 2)
    plt.bar(range(len(dissimilarity_scores)), dissimilarity_scores, color="green")
    plt.title("Dissimilarity Scores for Generated Stories")
    plt.xlabel("Sample Index")
    plt.ylabel("Dissimilarity Score (1 - Similarity)")

    plt.tight_layout()
    plt.savefig("/scratch/s5107318/LTP/LTP_Project/Results/metrics_plot.png")
    plt.close()

if __name__ == "__main__":
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

    model = StoryGenerationModel()
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
