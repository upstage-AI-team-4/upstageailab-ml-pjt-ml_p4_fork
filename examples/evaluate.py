import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.utils.evaluator import ModelEvaluator
from src.config import Config
from src.data.nsmc_dataset import NSMCDataModule
from src.utils.mlflow_utils import MLflowModelManager
from transformers import AutoTokenizer

def load_model_and_tokenizer(config):
    """Load production model and tokenizer"""
    model_manager = MLflowModelManager(config)
    
    # Get production model info
    model_info = model_manager.load_production_model_info()
    if model_info is None:
        raise RuntimeError("No production model found. Please train and promote a model first.")
    
    print("\n=== Loading Production Model ===")
    print(f"Model: {model_info['run_name']}")
    print(f"Metrics: {model_info['metrics']}")
    print(f"Stage: {model_info['stage']}")
    print(f"Timestamp: {model_info['timestamp']}")
    
    # Load model
    model = model_manager.load_production_model(config.project['model_name'])
    if model is None:
        raise RuntimeError("Failed to load the model. Please check if the model files exist.")
    
    # Load tokenizer based on model info
    tokenizer = AutoTokenizer.from_pretrained(model_info['params']['pretrained_model'])
    
    return model, tokenizer, model_info

def load_data_module(config, tokenizer):
    """Load data module"""
    data_module = NSMCDataModule(
        tokenizer=tokenizer,
        batch_size=config.base_training.batch_size,
        max_length=config.base_training.max_length,
        sampling_rate=config.data.sampling_rate,
        config=config,
        data_dir=str(config.data_path),
        train_file=config.data.train_data_path,
        val_file=config.data.val_data_path
    )
    data_module.prepare_data()
    data_module.setup(stage='fit')
    return data_module

def print_evaluation_results(metrics, model_info):
    """Print evaluation results with model info"""
    print("\n=== Model Information ===")
    print(f"Model Name: {model_info['run_name']}")
    print(f"Stage: {model_info['stage']}")
    print(f"Training Metrics: {model_info['metrics']}")
    print(f"Timestamp: {model_info['timestamp']}")
    
    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Average Confidence: {metrics['avg_confidence']:.4f}")
    
    print("\n=== Accuracy by Confidence Level ===")
    for bin_name, bin_data in metrics['confidence_bins'].items():
        print(f"{bin_name}: {bin_data['accuracy']:.4f} ({bin_data['count']} samples)")
    
    print("\n=== Sample Predictions ===")
    for sample in metrics['sample_predictions']:
        print(f"Text: {sample['text']}")
        print(f"True Label: {'Positive' if sample['true_label'] == 1 else 'Negative'}")
        print(f"Predicted: {'Positive' if sample['predicted_label'] == 1 else 'Negative'} (confidence: {sample['confidence']:.4f})")
        print(f"Correct: {'✓' if sample['true_label'] == sample['predicted_label'] else '✗'}")
        print("-" * 80)

def main():
    config = Config()
    
    try:
        # Load production model and tokenizer
        model, tokenizer, model_info = load_model_and_tokenizer(config)
        
        # Load data module
        data_module = load_data_module(config, tokenizer)
        
        # Evaluate model
        evaluator = ModelEvaluator(model, tokenizer)
        metrics = evaluator.evaluate_dataset(data_module)
        
        # Print results
        print_evaluation_results(metrics, model_info)
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 