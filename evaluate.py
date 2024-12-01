from src.utils.evaluator import ModelEvaluator
from src.utils.config import Config
from src.models.kcbert_model import KcBERT
from src.data.nsmc_dataset import NSMCDataModule
from transformers import AutoTokenizer

def load_model(config):
    model = KcBERT(**config.get_model_kwargs())
    return model

def load_tokenizer(config):
    return AutoTokenizer.from_pretrained(config.base_training.pretrained_model)

def load_data_module(config):
    tokenizer = load_tokenizer(config)
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

def print_evaluation_results(metrics):
    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Average Confidence: {metrics['avg_confidence']:.4f}")
    print("\n=== Accuracy by Confidence Level ===")
    for bin_name, bin_data in metrics['confidence_bins'].items():
        print(f"{bin_name}: {bin_data['accuracy']:.4f} ({bin_data['count']} samples)")
    print("\n=== Sample Predictions ===")
    for sample in metrics['sample_predictions']:
        print(f"Text: {sample['text']}")
        print(f"True Label: {sample['true_label']}")
        print(f"Predicted: {sample['predicted_label']} (confidence: {sample['confidence']:.4f})")
        print(f"Correct: {'✓' if sample['correct'] else '✗'}")
        print("-" * 80)

def main():
    config = Config()
    
    model = load_model(config)
    tokenizer = load_tokenizer(config)
    data_module = load_data_module(config)
    
    evaluator = ModelEvaluator(model, tokenizer)
    metrics = evaluator.evaluate_dataset(data_module)
    
    print_evaluation_results(metrics)

if __name__ == '__main__':
    main() 