import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tempfile
import os

def plot_confusion_matrix(y_true, y_pred, labels=['Negative', 'Positive'], normalize=False, log_path=None):
    """Confusion matrix 생성 및 저장
    
    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        labels: 레이블 이름
        normalize: 정규화 여부
        log_path: 저장 경로 (None이면 임시 파일로 저장)
        
    Returns:
        저장된 파일 경로
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 6))
    
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap,
                xticklabels=labels, yticklabels=labels, cbar=False)
    
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''), fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    
    if log_path is None:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            plt.savefig(tmp.name)
            plt.close()
            return tmp.name
    else:
        plt.savefig(log_path)
        plt.close()
        return log_path