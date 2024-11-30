import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tempfile
import os

def plot_confusion_matrix(y_true, y_pred, labels=['Negative', 'Positive']):
    """Confusion matrix 생성 및 임시 파일로 저장"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        plt.savefig(tmp.name)
        plt.close()
        return tmp.name 