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


## Creating and activating the environment

### Requirements

To run this project, you need to set up a Conda environment with the necessary dependencies.

### Conda Installation

Ensure you have **Conda** installed on your system. You can use either **Miniconda** (lightweight version) or **Anaconda** (full distribution). 

- [Miniconda Installation](https://docs.conda.io/en/latest/miniconda.html)
- [Anaconda Installation](https://www.anaconda.com/products/distribution)

Once **Conda** is installed, you can create and activate the environment as follows.

### Create the envirnonment
``` bash
conda env create --file environment.yml
```

### Activate the environment:
``` bash
conda activate ltp_env
```
