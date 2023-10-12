# Liger Label Model Testing

Using the Liger object to expand programmatic labeling functions with the help of Large Language Models (LLMs) embeddings.

## Introduction

This repository aims to test the performance of the Liger labeling functions expansion technique using LLMs. In the main repository, i.e. [Liger](https://github.com/HazyResearch/liger), The Liger object expands the labeling functions only in binary classification tasks and uses the [FlyingSquid](https://github.com/HazyResearch/flyingsquid) framework to train the label model. We tend to use the Liger object to expand labeling functions in a _Fake News Classification Task_ and train the [Snorkel](https://github.com/snorkel-team/snorkel) label model using the expanded labeling functions instead of the FlyingSquid.

Note: In the Liger object, **0** is used as the abstain signal. We will change it to **-1** to be compatible with the Snorkel label model.

For more information, refer to the paper [Shoring Up the Foundations Fusing Model Embeddings and Weak Supervision](https://arxiv.org/abs/2203.13270).

## NLP Task

We tested the performance of the Liger labeling functions expansion in the Fake News binary Text Classification task on the [FNID](https://ieee-dataport.org/open-access/fnid-fake-news-inference-dataset) dataset. We used the already written labeling functions from Chapter 03 of the book [Practical Weak Supervision with Snorkel](https://github.com/practicalweaksupervisionbook/companion).
In this task, we used **1** as the _REAL_ class, **0** as the _FAKE_ class, and **-1** as the _ABSTAIN_ class. We applied the labeling functions on the training and validation dataset and saved the results as raw, without expansion, matrices.

## Raw Labeling Functions

We trained a Snorkel label model on the raw labeling functions matrices and tuned the L2 regularization parameter to maximize the **recall score** of the label model. Then, we used the trained label model to weakly label the training and validation datasets.

The weakly labeled dataset was used to train an end model and the weighted **F1 score** on the _gold test_ set was **84**.

## Expanded Labeling Functions

Two embedding models were used as feature extractors for the Liger object to expand the raw labeling functions.

### Cohere API

First, we utilized the [Cohere](https://cohere.com/) Embedding endpoint to extract the embeddings of the training and validation sentences. We batched the embedding calls to work below the limit of the free plan (200 calls per minute). We, then, compared the coverage and performance of the expanded and raw labeling functions in the [Compare Coverage](./compare_coverage.ipynb) notebook. The expanded labeling functions matrices were then used to train the Snorkel labeling model which then was utilized to weakly label the training and validation datasets. The performance of the end model increased by **1** point weighted **F1 score** to be **85**.

### Llama 7B

Recently, [Llama](https://huggingface.co/spaces/meta-llama/README) models gained a lot of fame to being more reliable and performant. We used the [Llama-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) model as a sentence encoder via the [Sentence-Transformers](https://www.sbert.net/) framework. Again, the coverage comparison between the expanded and raw labeling functions is reported in the [Compare Coverage](./compare_coverage.ipynb) notebook. The Llama 7B embeddings increased the weighted **F1 score** on the gold test set by **3** points to be **87**.

## Conclusion

The approach of expanding the labeling functions using the nearest neighbors based on the embeddings from LLMs has indeed increased the performance of both the label model and the end model. While this is done with a similarity threshold of 0.85 for all labeling functions and both the embedding models, we believe tuning the thresholds will definitely increase the performance of both the label model and the end model.
