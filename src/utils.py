import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import os

def get_project_root():
    """Get the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)

def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """
    Plot confusion matrix for a model.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Genuine', 'Fraud'],
                yticklabels=['Genuine', 'Fraud'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path)
    plt.close()

def save_models(models_dict, filepath=None):
    """
    Save trained models to disk.
    
    Args:
        models_dict: Dictionary of model names and model objects
        filepath: Path to save the models (default: models/saved_models.pkl relative to project root)
    """
    if filepath is None:
        project_root = get_project_root()
        filepath = os.path.join(project_root, 'models', 'saved_models.pkl')
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    joblib.dump(models_dict, filepath)
    print(f"Models saved to {filepath}")

def load_models(filepath=None):
    """
    Load trained models from disk.
    
    Args:
        filepath: Path to load the models from (default: models/saved_models.pkl relative to project root)
    
    Returns:
        Dictionary of model names and model objects
    """
    if filepath is None:
        project_root = get_project_root()
        filepath = os.path.join(project_root, 'models', 'saved_models.pkl')
    return joblib.load(filepath)

def classification_report_dataframe(y_true, y_pred, model_name):
    """
    Generate classification report as DataFrame.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
    
    Returns:
        DataFrame with classification metrics
    """
    report = classification_report(y_true, y_pred, output_dict=True, target_names=['Genuine', 'Fraud'])
    df = pd.DataFrame(report).transpose()
    df['Model'] = model_name
    return df

