from evaluate import load
from pathlib import Path
import os
import sys
import logging
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from torch.profiler import profile, ProfilerActivity
from sentence_transformers import SentenceTransformer, util
from torch.distributed import destroy_process_group
from concurrent.futures import ThreadPoolExecutor
from language_tool_python import LanguageTool

# Import custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from Modules.flan_model import FlanT5Model
from Modules.data_loader import DataLoader as dl


# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# language tool setup
tool = LanguageTool("en-US")

# Logger setup
root = Path(__file__).resolve().parent.parent
log_dir = root / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
logs_path = log_dir / 'cot_wrapper_info.log'

if not log_dir.exists():
    print(f"Log directory {log_dir} was not created successfully.")
else:
    print(f"Log directory exists: {log_dir}")

# Debugging log file path
print(f"Logging to file: {logs_path}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cot_wrapper_info.log", mode="w"),
        logging.StreamHandler(sys.stdout)
    ]
)

class Logger:
    @staticmethod
    def log_info(message):
        logging.info(message)

    @staticmethod
    def log_error(message):
        logging.error(message)

COT_PROMPT = (
    "Emotion: {emotion}\n"
    "Utterance: {input_description}\n"
    "Art Style: {art_style}\n"
    "Write a vivid and engaging story inspired by the above. "
    "Make sure the story reflects the given emotion, incorporates elements of the art style, "
    "and addresses the key elements of the description provided. Use a narrative format to tell a cohesive and creative story."
)

Logger.log_info("Test log message to verify logging setup.")


class CoTWrapper:
    def __init__(self, baseline_model: FlanT5Model) -> None:
        Logger.log_info("Initializing CoT prompting.")
        self.baseline_model = baseline_model
        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

    def preprocess_dataset(self, dataset):
        max_length = 512

        def tokenize_function(batch):
            prompts = [
                COT_PROMPT.format(
                    emotion=emotion,
                    input_description=utterance,
                    art_style=art_style
                )
                for emotion, utterance, art_style in zip(
                    batch["emotion"], batch["utterance"], batch["art_style"]
                )
            ]

            inputs = self.baseline_model.tokenizer(
                prompts,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="np"
            )
            targets = self.baseline_model.tokenizer(
                batch["target_output"],
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="np"
            )

            return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "labels": targets["input_ids"]
            }

        Logger.log_info("Tokenizing dataset...")
        return dataset.map(
            tokenize_function,
            batched=True,
            batch_size=16,
            remove_columns=dataset.column_names
        )

    def clean_generated_text(self, text):
        sentences = text.split(". ")
        cleaned = ". ".join(list(dict.fromkeys(sentences)))
        corrected = tool.correct(cleaned)
        return corrected

    def generate_with_cot(self, emotion: str, utterance: str, art_style: str, min_tokens=100, max_tokens=200) -> str:
        prompt = COT_PROMPT.format(emotion=emotion, input_description=utterance, art_style=art_style)
        raw_story = self.baseline_model.generate_story(prompt, min_tokens=min_tokens, max_tokens=max_tokens)
        return self.clean_generated_text(raw_story)

    def evaluate_with_cot(self, val_dataset) -> dict:
        Logger.log_info("Evaluating the model using CoT prompting.")
        bertscore = load("bertscore")

        def process_example(example):
            try:
                emotion = example["emotion"]
                utterance = example["utterance"]
                art_style = example["art_style"]
                reference_story = example["target_output"]
                generated_story = self.generate_with_cot(emotion, utterance, art_style)
                return generated_story, reference_story
            except KeyError as e:
                Logger.log_error(f"Missing key in dataset: {e}")
                return None

        with ThreadPoolExecutor() as executor:
            results = list(filter(None, executor.map(process_example, val_dataset)))

        predictions = [res[0] for res in results]
        references = [res[1] for res in results]

        # Batch similarity computations
        generated_encodings = self.sbert_model.encode(predictions, convert_to_tensor=True)
        reference_encodings = self.sbert_model.encode(references, convert_to_tensor=True)
        dissimilarity_scores = (1 - util.cos_sim(generated_encodings, reference_encodings).diagonal()).tolist()

        Logger.log_info("Computing the BERT and Dissimilarity scores...")
        bert_score = bertscore.compute(predictions=predictions, references=references, lang="en")
        average_bert_score = sum(bert_score["f1"]) / len(bert_score["f1"])
        average_dissimilarity = sum(dissimilarity_scores) / len(dissimilarity_scores)

        Logger.log_info(f"Overall BERTScore F1: {average_bert_score:.4f}")
        Logger.log_info(f"Overall Dissimilarity: {average_dissimilarity:.4f}")

        print(f"Overall BERTScore F1: {average_bert_score:.4f}")
        print(f"Overall Dissimilarity: {average_dissimilarity:.4f}")

        self.plot_metrics(bert_score["f1"], dissimilarity_scores)

        return {"BERTScore_F1": bert_score["f1"], "Dissimilarity": dissimilarity_scores}

    def plot_metrics(self, bert_scores, dissimilarity_scores):
        Logger.log_info("Plotting the results...")
        try:
            # bar plot for bertscore
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(bert_scores)), bert_scores, color="blue", label="BERTScore F1")
            plt.title("BERT Scores for Generated Stories")
            plt.savefig(root / "Results" / "cot_metrics_bert.png")
            plt.close()

            # bar plot for dissimilarity
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(dissimilarity_scores)), dissimilarity_scores, color="green", label="Dissimilarity (1 - Similarity)")
            plt.legend()
            plt.title("Dissimilarity Scores for Generated Stories")
            plt.savefig(root / "Results" / "cot_metrics_dissimilarity.png")
            plt.close()

            # scattered plot
            plt.figure(figsize=(10, 6))
            plt.scatter(dissimilarity_scores, bert_scores, alpha=0.7, edgecolor='k')
            plt.title("BERTScore F1 vs. Dissimilarity", fontsize=14)
            plt.xlabel("Dissimilarity (1 - Similarity)", fontsize=12)
            plt.ylabel("BERTScore F1", fontsize=12)
            plt.grid(True)
            plt.savefig(root / "Results" / "cot_metrics_scattered.png")
            plt.close()

            # heatmap
            data = pd.DataFrame({"BERTScore F1": bert_scores, "Dissimilarity": dissimilarity_scores})
            sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Correlation Between Metrics", fontsize=14)
            plt.savefig(root / "Results" / "cot_metrics_heatmap.png")
            plt.close()

            # line plot
            sorted_indices = np.argsort(bert_scores)
            sorted_bert_scores = [bert_scores[i] for i in sorted_indices]
            sorted_dissimilarity_scores = [dissimilarity_scores[i] for i in sorted_indices]

            plt.figure(figsize=(10, 6))
            plt.plot(sorted_bert_scores, label="BERTScore F1", marker='o')
            plt.plot(sorted_dissimilarity_scores, label="Dissimilarity", marker='x')
            plt.title("Metrics Sorted by BERTScore F1", fontsize=14)
            plt.xlabel("Samples (sorted)", fontsize=12)
            plt.ylabel("Score", fontsize=12)
            plt.legend()
            plt.grid(True)
            plt.savefig(root / "Results" / "cot_metrics_line.png")
            plt.close()
            Logger.log_info("Successfully plotted the results in various formats.")

        except Exception as e:
            Logger.log_error(f"Error while plotting: {e}")


if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), '../Data/Artemis-cleaned.csv')

    Logger.log_info("Loading dataset...")
    raw_dataset = dl.load_artemis_dataset(file_path)

    Logger.log_info("Splitting the dataset into 70% training and 30% testing sets.")
    dataset_split = raw_dataset.train_test_split(test_size=0.3)
    train_dataset, val_dataset = dataset_split["train"], dataset_split["test"]

    baseline_model = FlanT5Model(model_name="google/flan-t5-small")
    cot_wrapper = CoTWrapper(baseline_model)

    Logger.log_info("Preprocessing datasets...")
    train_df = cot_wrapper.preprocess_dataset(train_dataset)
    val_df = cot_wrapper.preprocess_dataset(val_dataset)

    Logger.log_info("Training the model...")
    cot_wrapper.baseline_model.train(train_df, val_df, epochs=5, batch_size=16)

    Logger.log_info("Evaluating the model with 10 samples limit...")
    subset_val = val_dataset.select(range(10))
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')) as prof:
        metrics = cot_wrapper.evaluate_with_cot(subset_val)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    Logger.log_info(f"Final Metrics: {metrics}")

    Logger.log_info("Generating stories for three samples...")
    for i, example in enumerate(val_dataset.select(range(3))):
        emotion = example["emotion"]
        utterance = example["utterance"]
        art_style = example["art_style"]
        generated_story = cot_wrapper.generate_with_cot(emotion, utterance, art_style)
        Logger.log_info(f"Sample {i + 1}: Emotion: {emotion}, Utterance: {utterance}, Art Style: {art_style}, Generated: {generated_story}")
        print(f"\nSample {i + 1}:")
        print(f"Emotion: {emotion}")
        print(f"Utterance: {utterance}")
        print(f"Art Style: {art_style}")
        print(f"Generated Story: {generated_story}")

    # cleanup
    if torch.distributed.is_initialized():
        destroy_process_group()
