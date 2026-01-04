# [Sentiment Analysis Approaches - SST2]

## Overview
The goal of the project was to compare different approaches to binary sentiment classification on the Standford Sentiment Treebank v2. Approaches included traditional text mining approach, finetuning transformer approach, and LLM API call approach.

## Data
- Hugging Face Dataset consisting of 70,000 phrases from movie reviews and their accompanying classification.
- Binary classification: 0 - NEGATIVE or 1 - POSITIVE
- https://huggingface.co/datasets/stanfordnlp/sst2

## Methods
### Text Mining
- Trained SVM and Logistic Regression model to predict binary sentiment of phrases
- Used spaCy and TF-IDF to tokenize text and create word embeddings with TF-IDF
- Libraries: spaCy, Numpy, datasets, evaluate, sklearn

### Finetuning transformer
- Chose distilbert base uncased model to finetune on SST2 to perform binary sentiment classification
- Established hyperparameters with training_arguments and created trainer object to finetune model
- After trial and error, successfully created finetuned transformer using kaggle GPU after five training epochs
- Libraries: transformers, datasets, torch, numpy, evaluate

### GPT 5.1 Calls
- Created function to make calls to GPT model to perform sentiment analysis tasks
- Established streamlined system prompt and user prompt to classify sentiment as NEGATIVE or POSITIVE
- NOTE: Due to limitations, only tested 100 sampled phrases
- Libraries: openai, json, datasets, numpy, evaluate

## Results
- Text Mining: SVM achieved 88.89% accuracy on ~6000 test phrases
- Transformer: Custom distilbert uncased achieved 94.71% accuracy on ~6000 test phrases
- GPT 5.1: Achieved 91% accuracy on 100 test phrases

## Key Learnings
- Finetuning a transformer model was vastly superior to the text mining approach, with the tradeoff being training time. Without using a GPU, the training time would be approximately 8 - 10 hours. Unfortunately, due to the rate limits on OpenAI calls, I was unable to test a large sample size with GPT 5.1. I am unsure how successful this approach will be
- My transformer had only three training runs. With more robust optimization to hyperparamters, I believe ~2-3% accuracy improvements could be made
