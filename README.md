# Generating Emotionally Rich and Abstract Narratives from Visual Input Descriptions Using LLMs

## Authors: Janan Jahed, Alexandru Cernat & Andrei Medesan

### Group 4

Steps to consider when developing the code repository:

1. Download the ArtEmis dataset (either through an API or external download).

2. Preprocessing of the ArtEmis dataset as highlighted in the methods section.

3. Incorporate all LLM models (the one fine-tuned and baseline models).

3. Fine-tuning LLM by downloading the pre-trained model and prepare it for training on the dataset (this includes training the emotion embedding layer alongside the LLM’s text generation layers).

4. Develop a narrative generation pipeline inspired by CoT prompting to ensure the LLM captures emotional and contextual nuances in storytelling.

5. Evaluation framework implementaion (e.g. BLEU or ROUGE)

6. Compare results with baselines outputs (e.g. non-fine-tuned LLM narratives).