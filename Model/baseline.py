import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from Modules.data_loader import DataLoader as dl
from evaluate import load
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import logging

class FlanT5Model:
    def __init__(self, model_name="google/flan-t5-small"):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()

    def generate_story(self, input_prompt, emotion="neutral"):
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

    def _ensure_paragraph_length(self, story, max_lines=5):
        lines = story.split('. ')
        if len(lines) > max_lines:
            story = '. '.join(lines[:max_lines]) + '.'
        elif len(lines) < max_lines:
            story += ' ' + ' '.join(['...'] * (max_lines - len(lines)))
        return story

    def evaluate_model(self, val_dataset):
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
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings1 = model.encode([text1], convert_to_tensor=True)
        embeddings2 = model.encode([text2], convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        return cosine_scores.item()

    def _plot_results(self, scores, dissimilarity_scores):
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

if __name__ == "__main__":
    log_file_path = "/scratch/s5107318/LTP/LTP_Project/Model/base_model_eval.log"
    logging.basicConfig(
        filename=log_file_path,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger()

    file_path = os.path.join(os.path.dirname(__file__), '../Data/Artemis-cleaned.csv')
    val_dataset = dl.load_artemis_dataset(file_path)

    model = FlanT5Model()
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
