import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
import pickle
import os

class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        
    def train(self, X_train, y_train):
        print("Training Logistic Regression model...")
        self.model.fit(X_train, y_train)
        print("\nTraining complete!")
        
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, filepath):
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)

class LSTMModel:
    def __init__(self, vocab_size=10000, embedding_dim=128, max_length=200):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token='<OOV>')
        self.model: tf.keras.Model = self.build_model()

    def prepare_sequences(self, texts, is_training=True):
        if is_training:
            self.tokenizer.fit_on_texts(texts)
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = tf.keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=self.max_length, padding='post', truncating='post'
        )
        
        return padded
    
    def build_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        print("Preparing sequences for LSTM...")
        X_train_seq = self.prepare_sequences(X_train, is_training=True)
        X_val_seq = self.prepare_sequences(X_val, is_training=False)
        
        print("\nLSTM model summary:")
        self.model.summary()
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True
        )
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'outputs/models/lstm_best.keras', save_best_only=True, monitor='val_accuracy'
        )
        
        print("Training LSTM model...")
        history = self.model.fit(
            X_train_seq, y_train,
            validation_data=(X_val_seq, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )
        return history
    
    def predict(self, texts):
        sequences = self.prepare_sequences(texts, is_training=False)
        predictions = self.model.predict(sequences)
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, texts):
        sequences = self.prepare_sequences(texts, is_training=False)
        return self.model.predict(sequences)
    
    def save(self, model_path, tokenizer_path):
        self.model.save(model_path)
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
    
    def load(self, model_path, tokenizer_path):
        self.model = tf.keras.models.load_model(model_path)
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)

def evaluate_model(y_true, y_pred, model_name):
    print(f"\n{'='*50}")
    print(f"{model_name} Results")
    print(f"{'='*50}")
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    return accuracy

def main():
    print("Loading processed data...")
    X_train_tfidf = np.load('data/processed/X_train_tfidf.npy')
    X_test_tfidf = np.load('data/processed/X_test_tfidf.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    X_train_text = pd.read_csv('data/processed/X_train.csv')['text']
    X_test_text = pd.read_csv('data/processed/X_test.csv')['text']

    X_train_text_lstm, X_val_text, y_train_lstm, y_val = train_test_split(
        X_train_text, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    os.makedirs('outputs/models', exist_ok=True)

    print("\n" + "="*50)
    print("LOGISTIC REGRESSION")
    print("="*50)
    
    lr_model = LogisticRegressionModel()
    lr_model.train(X_train_tfidf, y_train)
    lr_predictions = lr_model.predict(X_test_tfidf)
    lr_accuracy = evaluate_model(y_test, lr_predictions, "Logistic Regression")
    lr_model.save('outputs/models/logistic_regression.pkl')
    print(f"\nLogistic Regression model saved to outputs/models/logistic_regression.pkl")

    print("\n" + "="*50)
    print("LSTM MODEL")
    print("="*50)
    
    lstm_model = LSTMModel(vocab_size=10000, embedding_dim=128, max_length=200)
    history = lstm_model.train(X_train_text_lstm, y_train_lstm, X_val_text, y_val, epochs=10, batch_size=32)
    lstm_predictions = lstm_model.predict(X_test_text)
    lstm_accuracy = evaluate_model(y_test, lstm_predictions, "LSTM")
    lstm_model.save('outputs/models/lstm_model.keras', 'outputs/models/lstm_tokenizer.pkl')
    print(f"\nLSTM model and tokenizer saved to outputs/models/lstm_model.keras and outputs/models/lstm_tokenizer.pkl")
    
    os.makedirs('outputs/predictions', exist_ok=True)
    results_df = pd.DataFrame({
        'text': X_test_text,
        'true_label': y_test,
        'lr_prediction': lr_predictions,
        'lstm_prediction': lstm_predictions
    })
    results_df.to_csv('outputs/predictions/training_results.csv', index=False)
    print("\nTraining results saved to outputs/predictions/training_results.csv")

    print("\nTraining complete!")

if __name__ == "__main__":
    main()
