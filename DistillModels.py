from transformers import TrainingArguments
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np 
import evaluate 
metric = evaluate.load("accuracy")
accuracy_score = metric

clinc = load_dataset("clinc_oos", "plus")

def compute_metrics(pred):
 predictions, labels = pred
 predictions = np.argmax(predictions, axis=1)
 return accuracy_score.compute(predictions=predictions,
references=labels)


def tokenize_text(batch):
 return student_tokenizer(batch["text"], truncation=True)
class DistillationTrainingArguments(TrainingArguments):
 def __init__(self, *args, alpha=0.5, temperature=2.0,
    **kwargs):
    super().__init__(*args, **kwargs)
    8
    self.alpha = alpha
    self.temperature = temperature
class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs_stu = model(**inputs)
        # Extract cross-entropy loss and logits from student
        loss_ce = outputs_stu.loss
        logits_stu = outputs_stu.logits
        # Extract logits from teacher
        with torch.no_grad():
          outputs_tea = self.teacher_model(**inputs)
          logits_tea = outputs_tea.logits
          # Soften probabilities and compute distillation loss
        loss_fct = nn.KLDivLoss(reduction="batchmean")
        loss_kd = self.args.temperature ** 2 * loss_fct(
        F.log_softmax(logits_stu / self.args.temperature,
        dim=-1),
        F.softmax(logits_tea / self.args.temperature, dim=-1))
        # Return weighted student loss
        loss = self.args.alpha * loss_ce + (1. - self.args.alpha) *loss_kd
        return (loss, outputs_stu) if return_outputs else loss

student_ckpt = "distilbert-base-uncased"
student_tokenizer = AutoTokenizer.from_pretrained(student_ckpt)

clinc_enc = clinc.map(tokenize_text, batched=True, remove_columns=["text"])
clinc_enc = clinc_enc.rename_column("intent", "labels")
student_training_args = DistillationTrainingArguments(
 output_dir=finetuned_ckpt, evaluation_strategy = "epoch",
 num_train_epochs=5, learning_rate=2e-5,
 per_device_train_batch_size=batch_size,
 per_device_eval_batch_size=batch_size, alpha=1,
weight_decay=0.01,
 push_to_hub=True)
