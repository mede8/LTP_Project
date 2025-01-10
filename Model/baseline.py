import os
from Modules import FlanT5Model
from Modules import DataLoader as dl


if __name__ == "__main__":
    file_path = file_path = os.path.join(
        os.path.dirname(__file__), '../Data/Artemis.csv'
    )
    val_dataset = dl.load_artemis_dataset(file_path)
    model = FlanT5Model()

    print("Generating stories from the validation dataset...")
    for i, example in enumerate(val_dataset.select(range(1))):
        input_prompt = example["input_prompt"]
        reference_story = example["target_output"]
        generated_story = model.generate_story(input_prompt)
        print(f"\nSample {i + 1}:")
        print(f"Input Prompt: {input_prompt}")
        print(f"Reference Story: {reference_story}")
        print(f"Generated Story: {generated_story}")

    print("\nEvaluating the model...")
    metrics = model.evaluate_model(val_dataset)
    print("Evaluation Metrics:", metrics)
