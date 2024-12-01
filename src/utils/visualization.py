import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tempfile
import os
import torch
from PIL import Image
import io

def plot_confusion_matrix(dataset, model, tokenizer, labels=['Negative', 'Positive'], normalize=True):
    """Generate Confusion Matrix from model predictions
    
    Args:
        dataset: Dataset to evaluate
        model: Model to evaluate
        tokenizer: Tokenizer
        labels: Label names
        normalize: Whether to normalize values
    
    Returns:
        PIL Image object
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
    
    # Save image to memory
    buf = io.BytesIO()
    
    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
    
    plt.figure(figsize=(10, 8))
    
    # Use 'Blues' colormap for better visibility
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
    
    # Set font properties for better visibility
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 12
    
    plt.tight_layout()
    
    # Save image to memory and convert to PIL Image
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    buf.seek(0)
    image = Image.open(buf)
    
    return image