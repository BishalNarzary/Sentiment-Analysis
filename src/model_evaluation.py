import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score)
import pickle
import os

def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_roc_curve(y_true, y_proba, model_name, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_precision_recall_curve(y_true, y_proba, model_name, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_model_comparison(results_df, save_path):
    models = ['lr', 'lstm']
    model_names = ['Logistic Regression', 'LSTM']
    metrics = {
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': []
    }
    
    for model in models:
        y_pred = results_df[f'{model}_prediction']
        y_true = results_df['true_label']
        
        metrics['Accuracy'].append(accuracy_score(y_true, y_pred))
        metrics['Precision'].append(precision_score(y_true, y_pred))
        metrics['Recall'].append(recall_score(y_true, y_pred))
        metrics['F1-Score'].append(f1_score(y_true, y_pred))
    
    x = np.arange(len(model_names))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (metric_name, values) in enumerate(metrics.items()):
        ax.bar(x + i*width, values, width, label=metric_name)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim((0, 1.1))
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        for j, v in enumerate(values):
            ax.text(j + i*width, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    
    return pd.DataFrame(metrics, index=model_names)

def get_lstm_probabilities(X_test_text):
    lstm_model = tf.keras.models.load_model('outputs/models/lstm_best.keras')
    
    with open('outputs/models/lstm_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    sequences = tokenizer.texts_to_sequences(X_test_text)
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=200, padding='post', truncating='post'
    )
    
    proba = lstm_model.predict(padded, verbose=0)
    return proba.flatten()

def main():
    print("Loading results and data...")
    results_df = pd.read_csv('outputs/predictions/training_results.csv')
    y_true = results_df['true_label']
    X_test_text = results_df['text']

    print("Loading Logistic Regression model for probability predictions...")
    with open('outputs/models/logistic_regression.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    
    X_test_tfidf = np.load('data/processed/X_test_tfidf.npy')
    lr_proba = lr_model.predict_proba(X_test_tfidf)[:, 1]

    print("Loading LSTM model for probability predictions...")
    lstm_proba = get_lstm_probabilities(X_test_text)

    os.makedirs('outputs/visualizations', exist_ok=True)

    print("\nCreating confusion matrix visualizations...")
    plot_confusion_matrix(y_true, results_df['lr_prediction'], 'Logistic Regression', 'outputs/visualizations/confusion_matrix_lr.png')
    plot_confusion_matrix(y_true, results_df['lstm_prediction'], 'LSTM', 'outputs/visualizations/confusion_matrix_lstm.png')

    print("\nCreating ROC curves...")
    plot_roc_curve(y_true, lr_proba, 'Logistic Regression', 'outputs/visualizations/roc_curve_lr.png')
    plot_roc_curve(y_true, lstm_proba, 'LSTM', 'outputs/visualizations/roc_curve_lstm.png')

    print("\nCreating precision-recall curves...")
    plot_precision_recall_curve(y_true, lr_proba, 'Logistic Regression', 'outputs/visualizations/precision_recall_lr.png')
    plot_precision_recall_curve(y_true, lstm_proba, 'LSTM', 'outputs/visualizations/precision_recall_lstm.png')

    print("\nCreating model comparison...")
    comparison_df = plot_model_comparison(results_df, 'outputs/visualizations/model_comparison.png')
    print("\nModel Comparison:")
    print(comparison_df)
    comparison_df.to_csv('outputs/predictions/model_comparison.csv')
    print("\nModel comparison metrics saved to outputs/predictions/model_comparison.csv")

    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
    