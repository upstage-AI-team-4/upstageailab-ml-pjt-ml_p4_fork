import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tempfile
import os
import torch
from PIL import Image
import io

def plot_confusion_matrix(dataset, model, tokenizer, labels=['부정', '긍정'], normalize=True):
    """모델의 예측 결과로 Confusion Matrix 생성
    
    Args:
        dataset: 평가할 데이터셋
        model: 평가할 모델
        tokenizer: 토크나이저
        labels: 레이블 이름
        normalize: 정규화 여부
    
    Returns:
        PIL Image 객체
    """
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            inputs = {
                'input_ids': sample['input_ids'].unsqueeze(0).to(model.device),
                'attention_mask': sample['attention_mask'].unsqueeze(0).to(model.device)
            }
            
            outputs = model(**inputs)
            logits = outputs.logits
            pred_label = torch.argmax(logits, dim=-1).item()
            
            y_true.append(sample['labels'].item())
            y_pred.append(pred_label)
    
    # 이미지를 메모리에 저장
    buf = io.BytesIO()
    
    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
    
    plt.figure(figsize=(10, 8))
    
    # 일반적으로 많이 사용되는 Blues 색상맵 사용
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2%' if normalize else 'd',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        square=True
    )
    
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''),
             fontsize=14, pad=20)
    plt.ylabel('True Label', fontsize=12, labelpad=10)
    plt.xlabel('Predicted Label', fontsize=12, labelpad=10)
    
    plt.tight_layout()
    
    # 이미지를 메모리에 저장하고 PIL Image로 변환
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    buf.seek(0)
    image = Image.open(buf)
    
    return image