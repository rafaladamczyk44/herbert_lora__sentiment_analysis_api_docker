from typing import Union, Any, Optional

import torch
from torch import nn
from transformers import Trainer

class WeightedTrainer(Trainer):
    """https://discuss.huggingface.co/t/how-can-i-use-class-weights-when-training/1067/5"""
    def compute_loss(self, model, inputs, num_items_in_batch, return_outputs: bool = False):
        labels = inputs["labels"]
        # Forward
        outputs = model(**inputs)
        logits = outputs.get('logits')

        # Class weights inversely proportional to class frequencies
        # Class 0 (positive): 56.5% -> weight ~0.59
        # Class 1 (neutral): 13.0% -> weight ~2.56
        # Class 2 (negative): 30.5% -> weight ~1.09
        # Normalized weights for balanced learning
        class_weights = torch.tensor([0.59, 2.56, 1.09], device=model.device)

        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


