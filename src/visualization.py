import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_feature_importance(model, feature_names, title="Feature Importance", save_path=None):
    """
    Plot feature importance for tree-based models
    """
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        raise ValueError("Model does not have feature_importances_ attribute")
    
    # Sort features by importance
    indices = np.argsort(importances)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title(title)
    plt.xlabel('Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names=None, title="Confusion Matrix", save_path=None):
    """
    Plot confusion matrix for model predictions
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names if class_names else "auto",
                yticklabels=class_names if class_names else "auto")
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_model_comparison(model_names, accuracies, title="Model Comparison", save_path=None):
    """
    Plot comparison of model accuracies
    """
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    
    plt.bar(model_names, accuracies, color=colors)
    plt.title(title)
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.ylim(0.5, 1.0)
    
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_correlation_matrix(df, title="Correlation Matrix", save_path=None):
    """
    Plot correlation matrix for features
    """
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()