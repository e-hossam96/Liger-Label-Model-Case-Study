# Liger Label Model Testing

Using the Liger object to expand programmatic labeling functions with the help of Large Language Models (LLMs) embeddings.

## Introduction

This repository aims to test the performance of the Liger labeling functions expansion technique using LLMs. In the main repository, i.e. [Liger](https://github.com/HazyResearch/liger), The Liger object expands the labeling functions only in binary classification tasks and uses the [FlyingSquid](https://github.com/HazyResearch/flyingsquid) framework to train the label model. We tend to use the Liger object to expand labeling functions in a _Fake News Classification Task_ and train the [Snorkel](https://github.com/snorkel-team/snorkel) label model using the expanded labeling functions instead of the FlyingSquid.

Note: In the Liger object, **0** is used as the abstain signal. We will change it to be **-1** to be compatible with the Snorkel label model.

For more information, refer to the paper [Shoring Up the Foundations Fusing Model Embeddings and Weak Supervision](https://arxiv.org/abs/2203.13270).
