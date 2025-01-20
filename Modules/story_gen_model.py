from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from peft import get_peft_model, LoraConfig, TaskType
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import logging


class StoryGenerationModel:
    """
    A class for fine-tuning a story generation model using LoRA (Low-Rank Adaptation),
    generating stories, and evaluating the model's performance.
    """

    def __init__(self, model_name="google/flan-t5-small", log_file_path="../Logs/lora_training.log"):
        """
        Initializes the story generation model and sets up logging.

        Args:
            log_file_path (str): Path to the log file for recording events.
            model_name (str): The name of the pre-trained model from Hugging Face's model hub.
        """
        logging.basicConfig(
            filename=log_file_path,
            filemode="a",
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )
        self.logger = logging.getLogger()

        self.logger.info(f"Initializing model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def preprocess_dataset(self, dataset):
        """
        Tokenizes and processes a dataset for training or evaluation.

        Args:
            dataset: The dataset to be processed.

        Returns:
            The processed dataset with tokenized inputs and labels.
        """
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

        self.logger.info("Preprocessing dataset...")
        return dataset.map(
            tokenize_function, remove_columns=dataset.column_names
        )

    def _create_lora_model(self):
        """
        Creates and applies a LoRA configuration to the model.

        Returns:
            The model with the LoRA configuration applied.

        Raises:
            ValueError: If the LoRA configuration cannot be applied to the model.
        """
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=8,
            lora_alpha=32,
            target_modules=["q", "k", "v", "o", "wi", "wo"],
            lora_dropout=0.1,
        )
        try:
            return get_peft_model(self.model, lora_config)
        except ValueError as e:
            self.logger.error(f"Error in LoRA setup: {e}")
            self.logger.info("Inspecting model modules...")
            for name, _ in self.model.named_modules():
                self.logger.debug(name)
            raise

    def fine_tune(self, train_dataset, val_dataset, hyperparams):
        """
        Fine-tunes the model on the training dataset using LoRA.

        Args:
            train_dataset: The training dataset.
            val_dataset: The validation dataset.
            hyperparams (dict): A dictionary of hyperparameters, including 'learning_rate',
                'batch_size', and 'num_train_epochs'.
        """
        self.logger.info("Applying LoRA configuration...")
        self.model = self._create_lora_model()

        train_dataset = self.preprocess_dataset(train_dataset)
        val_dataset = self.preprocess_dataset(val_dataset)

        training_args = TrainingArguments(
            output_dir="../Results",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=hyperparams["learning_rate"],
            per_device_train_batch_size=hyperparams["batch_size"],
            num_train_epochs=hyperparams["num_train_epochs"],
            logging_dir="../Results/logs",
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

        self.logger.info("Starting training...")
        trainer.train()
        self.logger.info("Training completed.")

    def generate_story(self, input_prompt, emotion):
        """
        Generates a story based on an input prompt and emotion.

        Args:
            input_prompt (str): The prompt to guide story generation.
            emotion (str): The emotional tone of the story (e.g., 'happy', 'sad').

        Returns:
            str: The generated story.
        """
        self.logger.info(f"Generating story for input prompt: {input_prompt} with emotion: {emotion}")
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
        self.logger.info(f"Generated story: {story}")
        return story

    def _ensure_paragraph_length(self, story, max_lines=5):
        """
        Ensures that the story has a maximum of `max_lines` lines, truncating or padding if necessary.

        Args:
            story (str): The generated story text.
            max_lines (int): The maximum number of lines allowed in the story.

        Returns:
            str: The adjusted story text.
        """
        lines = story.split('. ')
        if len(lines) > max_lines:
            story = '. '.join(lines[:max_lines]) + '.'
        elif len(lines) < max_lines:
            story += ' ' + ' '.join(['...'] * (max_lines - len(lines)))
        return story

    def _compute_similarity(text1, text2):
        """
        Computes cosine similarity between two texts using SentenceTransformers.

        Args:
            text1 (str): The first text input.
            text2 (str): The second text input.

        Returns:
            float: The cosine similarity score.
        """
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings1 = model.encode([text1], convert_to_tensor=True)
        embeddings2 = model.encode([text2], convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        return cosine_scores.item()

    def _plot_results(scores, dissimilarity_scores):
        """
        Plots evaluation results, including BERTScore F1 and dissimilarity scores.

        Args:
            scores (dict): A dictionary containing BERTScore F1 values.
            dissimilarity_scores (list): A list of dissimilarity scores.

        Saves:
            The plot as an image file in the specified directory.
        """
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
        plt.savefig("../Results/metrics_plot.png")
        plt.close()
