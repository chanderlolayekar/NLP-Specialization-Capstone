# Natural Language Processing Specialization — Capstone Project

A comprehensive, production-ready portfolio repository implementing core models, algorithms, and architectures from the **DeepLearning.AI Natural Language Processing Specialization**.

This repository bridges foundational mathematical concepts—implemented from scratch using NumPy—with modern deep learning workflows in TensorFlow and Hugging Face Transformers.

---

## Table of Contents
- [Project Architecture & Modules](#project-architecture--modules)
  - [Course 1: Classification & Vector Spaces](#course-1-classification--vector-spaces)
  - [Course 2: Probabilistic Models](#course-2-probabilistic-models)
  - [Course 3: Sequence Models](#course-3-sequence-models)
  - [Course 4: Attention Models & Transformers](#course-4-attention-models--transformers)
- [Repository Structure](#repository-structure)
- [Installation & Environment Setup](#installation--environment-setup)
- [How to Run Tests](#how-to-run-tests)
- [How to Use the Modules (Python Examples)](#how-to-use-the-modules-python-examples)

---

## Project Architecture & Modules

### Course 1: Classification & Vector Spaces
* **File:** `src/course_1_classification.py`
* **Concepts:** Feature Engineering, Word Embeddings, Linear Algebra Transformations, Cosine Similarity.
* **Implementations:**
  * **Word Analogies (`complete_analogy`):** Solves semantic relationships of the form $A : B :: C : D$ (e.g., *King : Man :: Queen : ?*) using vector arithmetic ($v_B - v_A + v_C$) and cosine similarity.
  * **Cross-Lingual Word Translation (`align_word_vectors`, `translate_word`):** Learns a transformation matrix $R$ using Gradient Descent to align vector spaces between two languages (e.g., English to Spanish).
  * **Text Classification (`NLPClassifier`):** Sentiment analysis baseline using Logistic Regression and Naïve Bayes.

### Course 2: Probabilistic Models
* **File:** `src/course_2_probabilistic.py`
* **Concepts:** Dynamic Programming, Markov Chains, Hidden Markov Models (HMM), Viterbi Decoding.
* **Implementations:**
  * **Minimum Edit Distance & Autocorrect (`min_edit_distance`, `autocorrect`):** Computes Levenshtein distance using dynamic programming to propose spelling corrections within a max edit threshold.
  * **Part-of-Speech Tagging (`viterbi_pos_tagger`):** Implements an HMM POS tagger utilizing the Viterbi algorithm to decode optimal hidden tag sequences from observation emissions.

### Course 3: Sequence Models
* **File:** `src/course_3_sequence.py`
* **Concepts:** Recurrent Neural Networks (RNNs), LSTMs, GRUs, Sequence Labeling, Contrastive Learning.
* **Implementations:**
  * **Named Entity Recognition (`build_ner_model`):** Sequence-to-sequence labeling utilizing Bidirectional LSTM/GRU layers with `TimeDistributed` Dense classification heads.
  * **Siamese Networks (`build_siamese_network`):** Dual twin subnetworks sharing weights to evaluate semantic similarity between sentence pairs via cosine distance.

### Course 4: Attention Models & Transformers
* **File:** `src/course_4_attention.py`
* **Concepts:** Self-Attention Mechanisms, Encoder-Decoder Architectures, Transfer Learning.
* **Implementations:**
  * **Scaled Dot-Product Attention (`scaled_dot_product_attention`):** NumPy implementation of $Softmax(\frac{QK^T}{\sqrt{d_k}})V$ with masking support.
  * **Transformer Pipeline Suite (`AttentionModels`):** Pretrained Hugging Face pipelines for Machine Translation, Text Summarization, and Question-Answering.

---

## Repository Structure

```text
NLP-Specialization-Capstone/
├── README.md                          # Project documentation and execution guide
├── requirements.txt                   # Dependency specifications
├── src/                               # Source modules for all 4 courses
│   ├── __init__.py
│   ├── course_1_classification.py     # Vector Spaces, Analogies, Word Translation
│   ├── course_2_probabilistic.py      # Edit Distance, Autocorrect, Viterbi POS Tagging
│   ├── course_3_sequence.py           # TensorFlow Bi-LSTM/GRU NER & Siamese Networks
│   └── course_4_attention.py          # Scaled Dot-Product Attention & Transformers
└── tests/                             # Unit tests for all modules
    ├── __init__.py
    ├── test_course_1.py
    ├── test_course_2.py
    ├── test_course_3.py
    └── test_course_4.py
# Running python -m unittest discover tests from your root folder will execute tests across all modules.
