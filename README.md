# Generating Emotionally Rich and Abstract Narratives from Visual Input Descriptions Using LLMs

This repository contains the code, models, and resources for the Language and Technology Practical project. The goal of this project is to explore and fine-tune large language models (LLMs) to create meaningful and emotionally resonant narratives from image descriptions.

## Problem Statement

Generating narratives that are emotionally rich and abstract from visual descriptions is a challenging task for LLMs. Existing models often struggle to capture emotional depth, maintain contextual relevance, and generate diverse outputs simultaneously. This project addresses these challenges by fine-tuning the Flan-T5 model using the ArtEmis dataset, which contains human-annotated emotional responses to visual art. We used the Low-Rank Adaptation (LoRA) fine-tuning and Chain-of-Thought (CoT) prompting to generate nuanced stories that balance creativity with coherence.

## Objectives

The primary objectives of this project are:
1. **Enhance Narrative Generation**: Fine-tune LLMs to produce emotionally and contextually rich narratives.
2. **Optimize Model Performance**: Explore and evaluate techniques like LoRA and CoT prompting to balance semantic alignment and diversity.
3. **Evaluate Outputs**: Use both quantitative (BERTScore, dissimilarity) and qualitative (human evaluation) methods to assess model performance.

## Results

- **Baseline Model**: Achieved a BERTScore of 0.90 and a dissimilarity score of 0.31, offering a balanced performance in terms of coherence and diversity.
- **LoRA Fine-Tuned Model**: Improved semantic alignment with a BERTScore of 0.91 but showed reduced diversity (dissimilarity score: 0.23).
- **CoT Fine-Tuned Model**: Generated the most diverse outputs (dissimilarity score: 0.57) with slightly reduced semantic alignment (BERTScore: 0.86).


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

## Navigating the repository

All the preprocessing and datasets can be found in the `Data/` folder. The `preprocessing.py` file preprocesses the Artemis.csv file by filetering out obscene content with the use of the JSON file `en.json`. The cleaned dataset is saved in the same folder as `Artemis-cleaned.csv` which is used to train and evaluate the models.

The `Model/` folder contains the necessary files to execute in order to obtain the results for the three models developed.
1. Baseline model in `baseline.py` file,
2. LoRA fine-tuned model in `lora.py` file,
3. CoT fine-tuned model in `cot_wrapper.py` file

To run these files, write the following in the terminal after activating the environment:
``` bash
python <model>.py
```
All these files use the models designed in the `Modules/` folder. Here, the `data_loader.py` file loads the Artemis dataset for the models, while `flan_model.py` and `story_gen_model.py` contain the necessary functions and classes for training and evaluating the models (baseline model and, respectively, the LoRA fine-tuned model). 

The results are saved as plots in the `Results/` folder.

## Contributors

- Janan Jahed
- Alexandru Cernat
- Andrei Medesan

## Acknowledgments

For creating this project we used the [Artemis Dataset](https://www.artemisdataset.org) and [Hugging Face](https://huggingface.co/models) for their tools and resources.
