from typing import Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use absolute path relative to this file
BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = str(BASE_DIR / 'models' / 'v5' / 'checkpoint-11475')
MODEL_ID = 'allegro/herbert-base-cased'

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=3,
        id2label={0: "positive", 1: "neutral", 2: "negative"},
        label2id={"positive": 0, "neutral": 1, "negative": 2}
    )
except Exception as e:
    print(f'Failed to load HF model {e}')
    raise


# Load LoRA adapters on top of base model
try:
      model = PeftModel.from_pretrained(base_model,CHECKPOINT_PATH)
      model.to(DEVICE)
      model.eval()
except Exception as e:
      print(f'Failed to load LoRA checkpoint: {e}')
      raise

def predict(text:str, max_len:int=512) -> Dict[str, float]:

    if text is None or len(text) == 0:
        return {
            "text": 'Provide a valid string input',
            'confidence': 0.0,
        }

    inputs = tokenizer(
        text,
        return_tensors='pt',
        max_length=max_len,
        truncation=True,
        padding=True,
    ).to(DEVICE)

    if inputs['input_ids'].shape[1]  > 512:
        return {
            'text': f'Input text too long, max length is {max_len}',
            'confidence': 0.0,
        }


    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_cls = logits.argmax(dim=-1).item()
        label = model.config.id2label[predicted_cls]
        probabilities = torch.softmax(logits, dim=-1)[0]
        confidence = probabilities[predicted_cls].item()
        return {
            "text": text,
            "label": label,
            "confidence": confidence,
            'probabilities': {
                'positive': probabilities[0].item(),
                'neutral': probabilities[1].item(),
                'negative': probabilities[2].item()
            }
        }