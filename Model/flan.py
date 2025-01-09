import os
import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from evaluate import load

def load_artemis_dataset(file_path):
    df = pd.read_csv(file_path)
    df["input_prompt"] = df["emotion"] + ": " + df["utterance"]
    df["target_output"] = df["utterance"]
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.1)
    return dataset["test"]

def generate_story(input_prompt, model, tokenizer):
    input_prompt = input_prompt + " Write a descriptive story about this scene in a few paragraphs."
    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
    outputs = model.generate(
        input_ids,
        max_new_tokens=500,
        min_length=500,       
        num_beams=5,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=2.0,
        no_repeat_ngram_size=3,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_model(val_dataset, model, tokenizer):
    rouge = load("rouge")
    bleu = load("bleu")
    predictions, references = [], []
    for example in val_dataset:
        input_prompt = example["input_prompt"]
        reference_story = example["target_output"]
        generated_story = generate_story(input_prompt, model, tokenizer)
        predictions.append(generated_story)
        references.append([reference_story])
    rouge_results = rouge.compute(predictions=predictions, references=references)
    bleu_results = bleu.compute(predictions=predictions, references=references)
    return {"ROUGE": rouge_results, "BLEU": bleu_results}

if __name__ == "__main__":
    file_path = os.path.expanduser("~/Desktop/LTP/LTP_Project/Data/Artemis.csv")
    val_dataset = load_artemis_dataset(file_path)
    model_name = "google/flan-t5-large"
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    print("Generating stories from the validation dataset...")
    for i, example in enumerate(val_dataset.select(range(1))):
        input_prompt = example["input_prompt"]
        reference_story = example["target_output"]
        generated_story = generate_story(input_prompt, model, tokenizer)
        print(f"\nSample {i + 1}:")
        print(f"Input Prompt: {input_prompt}")
        print(f"Reference Story: {reference_story}")
        print(f"Generated Story: {generated_story}")

    print("\nEvaluating the model...")
    metrics = evaluate_model(val_dataset, model, tokenizer)
    print("Evaluation Metrics:", metrics)
