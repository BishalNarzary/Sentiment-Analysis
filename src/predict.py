import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import os
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from data_preprocessing import TextPreprocessor
except ImportError:
    import importlib.util
    
    preprocessing_path = Path(__file__).resolve().parent / "data_preprocessing.py"
    spec = importlib.util.spec_from_file_location("data_preprocessing", preprocessing_path)
    
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load data_preprocessing module from {preprocessing_path}")
    
    data_preprocessing = importlib.util.module_from_spec(spec)
    sys.modules["data_preprocessing"] = data_preprocessing
    spec.loader.exec_module(data_preprocessing)
    TextPreprocessor = data_preprocessing.TextPreprocessor

class SentimentPredictor:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.lr_model = None
        self.lstm_model = None
        self.tfidf_vectorizer = None
        self.lstm_tokenizer = None
        
    def load_models(self):
        print("\nLoading models...")
        
        with open('outputs/models/logistic_regression.pkl', 'rb') as f:
            self.lr_model = pickle.load(f)
        
        with open('outputs/models/tfidf_vectorizer.pkl', 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        
        self.lstm_model = tf.keras.models.load_model('outputs/models/lstm_best.keras')
        
        with open('outputs/models/lstm_tokenizer.pkl', 'rb') as f:
            self.lstm_tokenizer = pickle.load(f)
        
        print("Models loaded successfully!")
    
    def predict_lr(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        cleaned_texts = [self.preprocessor.preprocess(text) for text in texts]
        X = self.tfidf_vectorizer.transform(cleaned_texts)                              # type: ignore
        predictions = self.lr_model.predict(X)                                          # type: ignore
        probabilities = self.lr_model.predict_proba(X)                                  # type: ignore         
        
        return predictions, probabilities
    
    def predict_lstm(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        cleaned_texts = [self.preprocessor.preprocess(text) for text in texts]
        sequences = self.lstm_tokenizer.texts_to_sequences(cleaned_texts)               # type: ignore
        padded = tf.keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=200, padding='post', truncating='post'
        )
        probabilities = self.lstm_model.predict(padded, verbose=0)                      # type: ignore
        predictions = (probabilities > 0.5).astype(int).flatten()
        
        return predictions, probabilities
    
    def predict(self, text, model='both'):
        results = {}
        
        if model in ['lr', 'both']:
            lr_pred, lr_proba = self.predict_lr(text)
            results['Logistic Regression'] = {
                'prediction': 'Positive' if lr_pred[0] == 1 else 'Negative',
                'confidence': float(lr_proba[0][lr_pred[0]])
            }
        
        if model in ['lstm', 'both']:
            lstm_pred, lstm_proba = self.predict_lstm(text)
            results['LSTM'] = {
                'prediction': 'Positive' if lstm_pred[0] == 1 else 'Negative',
                'confidence': float(lstm_proba[0][0]) if lstm_pred[0] == 1 else float(1 - lstm_proba[0][0])
            }
        
        return results

    def predict_batch(self, texts):
        print(f"\nRunning predictions on {len(texts)} reviews...")
        
        lr_preds, lr_probas = self.predict_lr(texts)
        lstm_preds, lstm_probas = self.predict_lstm(texts)

        results = []
        for i in range(len(texts)):
            results.append({
                'lr_prediction': 'Positive' if lr_preds[i] == 1 else 'Negative',
                'lr_confidence': float(lr_probas[i][lr_preds[i]]),
                'lstm_prediction': 'Positive' if lstm_preds[i] == 1 else 'Negative',
                'lstm_confidence': float(lstm_probas[i][0]) if lstm_preds[i] == 1 else float(1 - lstm_probas[i][0])
            })
        
        return pd.DataFrame(results)

def predict_unseen_data(predictor, filepath='data/raw/unseen_data.csv'):
    print(f"\nLoading unseen data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} reviews")
    print(f"\nTrue label distribution:\n{df['sentiment'].value_counts()}\n")

    pred_df = predictor.predict_batch(df['review'].tolist())

    label_map = {'negative': 'Negative', 'positive': 'Positive'}
    df['true_label'] = df['sentiment'].map(label_map)

    results_df = pd.concat([
        df[['review', 'true_label']].reset_index(drop=True),
        pred_df.reset_index(drop=True)
    ], axis=1)

    lr_correct = (results_df['true_label'] == results_df['lr_prediction']).sum()
    lstm_correct = (results_df['true_label'] == results_df['lstm_prediction']).sum()
    total = len(results_df)

    print("\n" + "="*70)
    print("UNSEEN DATA RESULTS")
    print("="*70)
    print(f"Logistic Regression Accuracy: {lr_correct/total:.4f} ({lr_correct}/{total})")
    print(f"LSTM Accuracy:                {lstm_correct/total:.4f} ({lstm_correct}/{total})")

    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS (first 5 reviews)")
    print("="*70)
    for _, row in results_df.head().iterrows():
        print(f"\nReview:     {row['review'][:80]}...")
        print(f"True Label: {row['true_label']}")
        print(f"LR:         {row['lr_prediction']} (confidence: {row['lr_confidence']:.4f})")
        print(f"LSTM:       {row['lstm_prediction']} (confidence: {row['lstm_confidence']:.4f})")

    os.makedirs('outputs/predictions', exist_ok=True)
    results_df.to_csv('outputs/predictions/prediction_results.csv', index=False)
    print("\nResults saved to outputs/predictions/prediction_results.csv")

    return results_df

def main():
    predictor = SentimentPredictor()
    predictor.load_models()

    predict_unseen_data(predictor)

if __name__ == "__main__":
    main()
    