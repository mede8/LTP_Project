from evaluate import load
from sentence_transformers import SentenceTransformer, util
from transformers import Trainer, TrainingArguments
import matplotlib.pyplot as plt


class FlanT5Model:
    """
    A class representing a Flan-T5 model for generating stories, 
    training on custom datasets, and evaluating the generated stories
    using metrics such as BERTScore and dissimilarity scores.
    """

    def __init__(self, model_name="google/flan-t5-small"):
        """
        Initializes the Flan-T5 model and tokenizer.

        Args:
            model_name (str): The name of the pre-trained Flan-T5 model from the Hugging Face model hub.
        """
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()

    def generate_story(self, input_prompt, emotion="neutral"):
        """
        Generates a detailed and emotional story based on the input prompt and desired emotion.

        Args:
            input_prompt (str): The input text prompt for generating a story.
            emotion (str): The emotional tone of the story.
        Returns:
            str: The generated story.
        """
        input_text = f"Write a detailed and emotional story that conveys {emotion}: {input_prompt}"
        inputs = self.tokenizer(
            input_text, return_tensors="pt", max_length=128, truncation=True, padding="max_length"
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
        return story

    def train(self, train_dataset, epochs, batch_size, output_dir="./flan_t5_trained"):
        """
        Trains the Flan-T5 model on the provided dataset.

        Args:
            train_dataset: The training dataset in a compatible Hugging Face Dataset format.
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training.
            output_dir (str): The directory where the trained model and logs will be saved.
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=500,
            save_total_limit=2,
            evaluation_strategy="steps",
            eval_steps=500,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset
        )

        trainer.train()

    def _ensure_paragraph_length(self, story, max_lines=5):
        """
        Ensures that the story is limited to a maximum number of lines by truncating or padding.
        Args:
            story (str): The story text to adjust.
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

    def evaluate_model(self, val_dataset):
        """
        Evaluates the Flan-T5 model using a validation dataset.

        Args:
            val_dataset: The validation dataset in a compatible Hugging Face Dataset format.

        Returns:
            dict: A dictionary containing the average BERTScore F1 and dissimilarity score.
        """
        predictions = []
        references = []
        dissimilarity_scores = []

        for i, example in enumerate(val_dataset.select(range(10))):
            input_prompt = example["input_prompt"]
            emotion = example.get("emotion", "neutral")
            reference_story = example["target_output"]
            generated_story = self.generate_story(input_prompt, emotion)

            predictions.append(generated_story)
            references.append(reference_story)
            input_similarity = self._compute_similarity(input_prompt, generated_story)
            dissimilarity_scores.append(1 - input_similarity)

        bertscore = load("bertscore")
        scores = bertscore.compute(predictions=predictions, references=references, lang="en")

        self._plot_results(scores, dissimilarity_scores)

        return {
            "bertscore_f1": sum(scores["f1"]) / len(scores["f1"]),
            "dissimilarity": sum(dissimilarity_scores) / len(dissimilarity_scores),
        }

    def _compute_similarity(self, text1, text2):
        """
        Computes the cosine similarity between two pieces of text using SentenceTransformers.

        Args:
            text1 (str): The first text input.
            text2 (str): The second text input.

        Returns:
            float: The cosine similarity score between the two inputs.
        """
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings1 = model.encode([text1], convert_to_tensor=True)
        embeddings2 = model.encode([text2], convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        return cosine_scores.item()

    def _plot_results(self, scores, dissimilarity_scores):
        """
        Plots the BERTScore F1 scores and dissimilarity scores as bar charts.

        Args:
            scores (dict): A dictionary containing BERTScore F1 values.
            dissimilarity_scores (list): A list of dissimilarity scores.

        Saves:
            The plot as an image file to the specified directory.
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
        plt.savefig("/scratch/s5107318/LTP/LTP_Project/Results/baseline_metrics_plot.png")
        plt.close()
