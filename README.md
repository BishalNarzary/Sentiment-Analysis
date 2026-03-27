# Sentiment Analysis

A machine learning project that classifies movie/product reviews as **positive** or **negative** using two models — Logistic Regression with TF-IDF features and a Bidirectional LSTM neural network.

----------------------------------------------------------------------------------------------------

## Overview

This project builds a binary sentiment classification pipeline that:
- Cleans and preprocesses raw text (lowercasing, URL removal, lemmatization, stopword filtering)
- Extracts TF-IDF features with bigram support for classical ML
- Trains a Logistic Regression model and a Bidirectional LSTM model
- Evaluates both models with accuracy, confusion matrices, ROC curves, and precision-recall curves
- Supports single-review and batch predictions on unseen data

----------------------------------------------------------------------------------------------------

## Project Structure

```
Sentiment-Analysis/
│
├── data/
│   ├── raw/
│   │   ├── review_data.csv                              # Labeled training reviews
│   │   └── unseen_data.csv                              # Unseen dataset for predictions
│   └── processed/
│       ├── X_train.csv                                  # Preprocessed training text + labels
│       ├── X_test.csv                                   # Preprocessed test text + labels
│       ├── X_train_tfidf.npy                            # TF-IDF training matrix
│       ├── X_test_tfidf.npy                             # TF-IDF test matrix
│       ├── y_train.npy                                  # Encoded training labels
│       └── y_test.npy                                   # Encoded test labels
│
├── outputs/
│   ├── models/
│   │   ├── lstm_best.keras                              # Best LSTM checkpoint
│   │   ├── lstm_model.keras                             # Final LSTM model
│   │   ├── lstm_tokenizer.pkl                           # LSTM word tokenizer
│   │   ├── tfidf_vectorizer.pkl                         # Fitted TF-IDF vectorizer
│   │   ├── logistic_regression.pkl                      # Trained LR model
│   │   └── label_mapping.json                           # Sentiment label encoding
│   ├── predictions/
│   │   ├── training_results.csv                         # LR & LSTM predictions on test set
│   │   ├── prediction_results.csv                       # Predictions on unseen data
│   │   └── model_comparison.csv                         # Metric comparison table
│   └── visualizations/
│       ├── confusion_matrix_lr.png                      # LR confusion matrix
│       ├── confusion_matrix_lstm.png                    # LSTM confusion matrix
│       ├── roc_curve_lr.png                             # LR ROC curve
│       ├── roc_curve_lstm.png                           # LSTM ROC curve
│       ├── precision_recall_lr.png                      # LR precision-recall curve
│       ├── precision_recall_lstm.png                    # LSTM precision-recall curve
│       └── model_comparison.png                         # Side-by-side model comparison chart
│
├── src/
│   ├── data_preprocessing.py                            # Text cleaning & TF-IDF feature creation
│   ├── model_training.py                                # LR & LSTM training
│   ├── model_evaluation.py                              # Metrics, plots & model comparison
│   └── predict.py                                       # Prediction interface (single & batch)
│
├── README.md                                            # Project documentation
└── requirements.txt                                     # Python dependencies
```

----------------------------------------------------------------------------------------------------

## Models

### Logistic Regression
- Input: TF-IDF vectors (3,000 features, unigrams + bigrams)
- Fast training, strong baseline for text classification

### Bidirectional LSTM
- Embedding layer → 2× Bidirectional LSTM layers → Dense layers
- Dropout (0.5) between layers to reduce overfitting
- Early stopping and best-model checkpointing during training

----------------------------------------------------------------------------------------------------

## Features

Text is preprocessed through the following pipeline before being fed to either model:

`Lowercasing`       —  Normalize case  
`URL & HTML removal`  —  Strip links and markup  
`Punctuation removal` —  Keep only alphabetic characters  
`Tokenization`      —  Split into word tokens  
`Lemmatization`     —  Reduce words to their base form  
`Stopword filtering` —  Remove common words (negations preserved: *not*, *no*, *never*, etc.)

**Target:** `sentiment` — binary label encoded as `0` (negative) or `1` (positive)

----------------------------------------------------------------------------------------------------

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/BishalNarzary/Sentiment-Analysis.git
cd Sentiment-Analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Preprocess the data
```bash
python src/data_preprocessing.py
```

### 4. Train the models
```bash
python src/model_training.py
```

### 5. Evaluate the models
```bash
python src/model_evaluation.py
```

### 6. Run predictions

**On unseen CSV data:**
```bash
python src/predict.py
```

----------------------------------------------------------------------------------------------------

## Data

The training data (`review_data.csv`) should contain at minimum two columns:

| Column      | Description                          |
|-------------|--------------------------------------|
| `review`    | Raw text of the review               |
| `sentiment` | Label — `positive` or `negative`     |

The unseen data (`unseen_data.csv`) follows the same format and is used to evaluate real-world performance after training.
