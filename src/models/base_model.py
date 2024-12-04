from abc import ABC, abstractmethod
import pytorch_lightning as pl
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Tuple, Any, Optional

class BaseModel(pl.LightningModule, ABC):
    """Base abstract class for all models."""
    
    @abstractmethod
    def forward(self, **kwargs) -> Any:
        """Forward pass logic."""
        pass
    
    @abstractmethod
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and scheduler."""
        pass
    
    @abstractmethod
    def _compute_loss(self, batch: Any) -> torch.Tensor:
        """Compute loss for a batch."""
        pass
    
    @abstractmethod
    def _compute_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
        """Compute metrics for predictions."""
        pass

class BaseTextClassifier(BaseModel):
    """Base class for text classification models."""
    
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
    
    def training_step(self, batch: Tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        loss, metrics = self._shared_step(batch, 'train')
        self.log_dict({f'train/{k}': v for k, v in metrics.items()})
        return {'loss': loss}
    
    def validation_step(self, batch: Tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        loss, metrics = self._shared_step(batch, 'val')
        self.log_dict({f'val/{k}': v for k, v in metrics.items()})
        return {'loss': loss}
    
    def _shared_step(self, batch: Tuple, split: str) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Shared step for training and validation."""
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        
        with torch.no_grad():
            preds = outputs.logits.argmax(dim=-1)
            metrics = self._compute_metrics(batch['labels'], preds)
            metrics['loss'] = loss.item()
            
        return loss, metrics