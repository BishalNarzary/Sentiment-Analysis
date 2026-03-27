import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import json
import os

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.negations = {'not', 'no', 'never', 'neither', 'nor', 'nobody', 'nothing', 'nowhere'}
        
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def tokenize_and_lemmatize(self, text):
        tokens = word_tokenize(text)
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if (token not in self.stop_words or token in self.negations) and len(token) > 2
        ]
        return ' '.join(tokens)
    
    def preprocess(self, text):
        text = self.clean_text(text)
        text = self.tokenize_and_lemmatize(text)
        return text

def load_and_preprocess_data(filepath, text_column='review', label_column='sentiment'):
    print("\nLoading data...")
    df = pd.read_csv(filepath)
    
    print(f"\nOriginal dataset shape: {df.shape}")
    print(f"Class distribution:\n{df[label_column].value_counts()}")
    
    preprocessor = TextPreprocessor()
    
    print("\nPreprocessing text...")
    df['cleaned_text'] = df[text_column].apply(preprocessor.preprocess)
    
    df = df[df['cleaned_text'].str.len() > 0]
    
    if df[label_column].dtype != 'int64':
        label_mapping = {'negative': 0, 'positive': 1}
        df['label'] = df[label_column].map(label_mapping)
        
        with open('outputs/models/label_mapping.json', 'w') as f:
            json.dump(label_mapping, f)
        
        print(f"\nLabel mapping: {label_mapping}")
        print(f"Label distribution after encoding:\n{df['label'].value_counts()}")
    else:
        df['label'] = df[label_column]
    
    return df

def create_train_test_split(df, test_size=0.2, random_state=42):
    X = df['cleaned_text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTrain set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

def create_tfidf_features(X_train, X_test, max_features=5000):
    print("\nCreating TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2, max_df=0.95)
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    with open('outputs/models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"TF-IDF vectorizer saved to outputs/models/tfidf_vectorizer.pkl")
    
    return X_train_tfidf, X_test_tfidf, vectorizer

def main():
    INPUT_FILE = 'data/raw/review_data.csv'

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    os.makedirs('outputs/models', exist_ok=True)
    df = load_and_preprocess_data(INPUT_FILE, text_column='review', label_column='sentiment')
    
    X_train, X_test, y_train, y_test = create_train_test_split(df)
    X_train_tfidf, X_test_tfidf, vectorizer = create_tfidf_features(X_train, X_test)
    
    print("\nSaving processed data...")
    os.makedirs('data/processed', exist_ok=True)
    
    pd.DataFrame({
        'text': X_train,
        'label': y_train
    }).to_csv('data/processed/X_train.csv', index=False)
    
    pd.DataFrame({
        'text': X_test,
        'label': y_test
    }).to_csv('data/processed/X_test.csv', index=False)
    
    np.save('data/processed/X_train_tfidf.npy', np.asarray(X_train_tfidf.todense()))
    np.save('data/processed/X_test_tfidf.npy', np.asarray(X_test_tfidf.todense()))
    np.save('data/processed/y_train.npy', y_train.values)
    np.save('data/processed/y_test.npy', y_test.values)
    
    print("\nPreprocessing complete!")

if __name__ == "__main__":
    main()
