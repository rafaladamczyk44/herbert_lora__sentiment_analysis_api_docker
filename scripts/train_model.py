from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Tokenize the text
def tokenize_function(examples):
    texts = [str(text) if text is not None else "" for text in examples["text"]]
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512
    )

model_id = 'allegro/herbert-base-cased'
num_labels = 3  # positive (0), neutral (1), negative (2)

model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=num_labels,
    id2label={0: "positive", 1: "neutral", 2: "negative"},
    label2id={"positive": 0, "neutral": 1, "negative": 2}
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

dataset = load_dataset(
    path='csv',
    data_files={
        'train': '../data/data_train.csv',
        'validation': '../data/data_validation.csv',
        'test': '../data/data_test.csv'
    }
)


dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.remove_columns(['text'])
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Compute metrics function for evaluation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,  # Sequence Classification for sentiment analysis
    r=16,                          # Rank: number of dimensions in LoRA matrices
    lora_alpha=32,                  # scaling
    target_modules=["query", "value"],  # HerBERT uses "query" and "value" (not "q", "v")
    lora_dropout=0.01,            # Dropout probability for LoRA layers
    bias="none",                  # Don't train bias parameters
    inference_mode=False          # Training mode
)

peft_model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir='../models',
    overwrite_output_dir=True,
    eval_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,

    # Gradient clipping
    max_grad_norm=1.0,

    # Weights & Biases integration
    report_to="wandb",  # Enable W&B logging
    run_name="herbert-lora-sentiment-v2",
    logging_steps=50,  # Log every 50 steps
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    compute_metrics=compute_metrics,  # Add metrics computation
)

trainer.train()

# Evaluate on test set
test_results = trainer.evaluate(dataset['test'])
print(f"\nTest Results:")
print(f"  Accuracy: {test_results.get('eval_accuracy', 0):.4f}")
print(f"  F1 Score: {test_results.get('eval_f1', 0):.4f}")
print(f"  Precision: {test_results.get('eval_precision', 0):.4f}")
print(f"  Recall: {test_results.get('eval_recall', 0):.4f}")