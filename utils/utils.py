def tokenize(examples, tokenizer):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=tokenizer.model_max_length, # 512
    )

def encode_labels(examples):
    label_mapping = {
        'positive': 0,
        'negative': 1,
        'neutral': 2,
    }
    return {'labels': [label_mapping[sentiment] for sentiment in examples['sentiment']]}