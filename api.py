from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import torch
from src.config import Config
from src.utils.mlflow_utils import MLflowModelManager
from transformers import AutoTokenizer

app = FastAPI(
    title="Sentiment Analysis API",
    description="Korean Sentiment Analysis API with multiple production models",
    version="1.0.0"
)

# Global variables for model caching
model_cache = {}
tokenizer_cache = {}
config = Config()
model_manager = MLflowModelManager(config)

class PredictionRequest(BaseModel):
    text: str
    model_id: str

class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    probabilities: List[float]
    num_tokens: int

def load_model_and_tokenizer(model_id: str):
    """Load model and tokenizer with caching"""
    if model_id not in model_cache:
        # Get model info
        model_info = None
        for model in model_manager.get_production_models():
            if model['run_id'] == model_id:
                model_info = model
                break
        
        if model_info is None:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Load model
        model = model_manager.load_production_model(config.project['model_name'])
        if model is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to load model"
            )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_info['params']['pretrained_model']
        )
        
        # Cache model and tokenizer
        model_cache[model_id] = model
        tokenizer_cache[model_id] = tokenizer
    
    return model_cache[model_id], tokenizer_cache[model_id]

@app.get("/models")
async def get_models() -> List[Dict[str, Any]]:
    """Get list of available production models"""
    models = model_manager.get_production_models()
    if not models:
        raise HTTPException(
            status_code=404,
            detail="No production models found"
        )
    return models

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict sentiment for given text using specified model"""
    try:
        model, tokenizer = load_model_and_tokenizer(request.model_id)
        
        # Tokenize
        inputs = tokenizer(
            request.text,
            padding='max_length',
            max_length=config.training_config['max_length'],
            truncation=True,
            return_tensors='pt'
        )
        
        # Check if model needs token_type_ids
        try:
            import inspect
            forward_params = inspect.signature(model.forward).parameters
            if 'token_type_ids' not in forward_params and 'token_type_ids' in inputs:
                del inputs['token_type_ids']
        except Exception as e:
            print(f"Warning: Error checking model signature: {e}")
            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']
        
        # Move inputs to model device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_label = torch.argmax(outputs.logits, dim=-1).item()
            confidence = probs[0][pred_label].item()
        
        return {
            "sentiment": "긍정" if pred_label == 1 else "부정",
            "confidence": confidence,
            "probabilities": probs[0].cpu().numpy().tolist(),
            "num_tokens": len(tokenizer.encode(request.text))
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 