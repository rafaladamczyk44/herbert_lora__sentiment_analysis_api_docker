from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch


model_id = 'allegro/herbert-base-cased'

tokenizer = AutoTokenizer.from_pretrained(model_id)

base_model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=3,
    id2label={0: "positive", 1: "neutral", 2: "negative"},
    label2id={"positive": 0, "neutral": 1, "negative": 2}
)

# Load LoRA adapters on top of base model
model = PeftModel.from_pretrained(base_model, '../models/v2/checkpoint-4485')

model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors='pt')
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