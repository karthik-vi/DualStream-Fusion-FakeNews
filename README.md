# Dual-Stream Feature Fusion Network for Fake News Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/transformers/)

This repository contains the code, datasets, and final report for our CSE 676 Deep Learning final project at the University at Buffalo. 

We propose and evaluate a **Dual-Stream Feature Fusion** framework for fake news detection that combines the dense contextual embeddings of pre-trained Large Language Models (RoBERTa) with explicit, handcrafted psycholinguistic and structural features. Our research specifically investigates the trade-offs of feature fusion on micro-texts and its ability to mitigate catastrophic domain shift during cross-dataset evaluation.

## 👥 Team Members
* **Sai Karthik Varma Indukuri** - Architecture Design & PyTorch Optimization
* **Srivatsa Vuppalanchi** - Experimental Design & Ablation Studies
* **Gandhar Sidhaye** - Data Engineering & Linguistic Feature Pipeline

## 🧠 Architecture Overview
Pure deep learning models often operate as "black boxes," analyzing semantic context while ignoring explicit psychological and stylistic markers of deception (e.g., hyperbole, anomalous readability). To address this, our architecture fuses two parallel streams:
1. **Contextual Stream:** A pre-trained `roberta-base` Transformer that extracts global semantic meaning via the `[CLS]` token embedding.
2. **Linguistic Feature Stream:** A custom NLP pipeline utilizing `TextBlob` and `textstat` to extract a continuous feature vector representing:
   * **Polarity** (Sentiment)
   * **Subjectivity** (Fact vs. Opinion)
   * **Readability** (Flesch-Kincaid Grade Level)
   * **Lexical Diversity** (Unique token ratio)

The vectors are normalized via `LayerNorm` to prevent structural data from dominating the contextual embeddings, then passed through a custom Multi-Layer Perceptron (MLP) classification head.

## 📊 Datasets
This project evaluates the model across two distinctly different paradigms:
* **In-Domain (Micro-Text):** The [LIAR Dataset](https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset), consisting of 12.8k short political statements and quotes.
* **Cross-Domain (Full Articles):** The [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) dataset, consisting of over 23k full-length web articles and gossip blogs. 

*Note: The dataset files required to run the notebook are included in this repository.*

## 🏆 Key Results & Findings

To isolate the impact of our custom linguistic stream, we conducted a rigorous ablation study comparing our Dual-Stream architecture against a standard RoBERTa-for-Sequence-Classification baseline.

| Metric | Dual-Stream (RoBERTa + Features) | RoBERTa-Only Baseline |
| :--- | :--- | :--- |
| **In-Domain (LIAR Test Set)** | 62.19% | **66.30%** |
| **Cross-Domain (FakeNewsNet)**| **33.47%** | 26.14% |

**Conclusions:**
1. **The Micro-Text Penalty:** Document-level linguistic features (like Readability) act as statistical noise when applied to short, 15-word political quotes, causing a slight penalty compared to pure semantic attention.
2. **Cross-Domain Robustness:** When subjected to extreme domain shift (evaluating a model trained on politics directly on web gossip), the standard Transformer collapsed entirely (26.14%). Our Dual-Stream architecture successfully anchored the model with domain-agnostic structural features, yielding a **7.3% performance retention** over the baseline.

## 🚀 How to Run the Code

The entire training, ablation study, and evaluation pipeline is contained within a single interactive Jupyter Notebook designed for Google Colab.
