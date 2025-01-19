from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Trainer, TrainingArguments
from evaluate import load


class FlanT5Model:
    def __init__(self, model_name="google/flan-t5-large"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def train(self, train_dataset, val_dataset, epochs,
              batch_size, output_dir="./flan_t5_trained"):
        """
        Train the Flan-T5 model on the train dataset.
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=500,
            save_total_limit=2,
            eval_strategy="steps",
            eval_steps=500,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            report_to="none",
            fp16=True,
            remove_unused_columns=False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        trainer.train()

    def test(self, test_dataset, batch_size):
        """
        Test the Flan-T5 model on the test dataset.
        """
        training_args = TrainingArguments(
            output_dir="./results",
            per_device_eval_batch_size=batch_size,
            logging_dir="./logs",
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            eval_dataset=test_dataset
        )

        results = trainer.evaluate()

        return results

    def generate_story(self, input_prompt, min_tokens, max_tokens):
        """
        Generate a descriptive story about a given scene.
        """
        input_prompt = input_prompt + " Write a descriptive story about this scene in a few paragraphs."
        input_ids = self.tokenizer(input_prompt, return_tensors="pt").input_ids

        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            min_length=min_tokens,
            num_beams=5,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=2.0,
            no_repeat_ngram_size=3,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def evaluate_model(self, val_dataset):
        """
        Evaluate the model using ROUGE and BLEU metrics.
        """
        rouge = load("rouge")
        bleu = load("bleu")
        predictions, references = [], []
        for example in val_dataset:
            input_prompt = example["input_prompt"]
            reference_story = example["target_output"]
            generated_story = self.generate_story(input_prompt)
            predictions.append(generated_story)
            references.append([reference_story])
        rouge_results = rouge.compute(predictions=predictions,
                                      references=references)
        bleu_results = bleu.compute(predictions=predictions,
                                    references=references)

        return {"ROUGE": rouge_results, "BLEU": bleu_results}
