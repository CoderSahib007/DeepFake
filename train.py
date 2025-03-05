# train.py
from transformers import Trainer, TrainingArguments
import torch

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).int()
    accuracy = (predictions == torch.tensor(labels)).float().mean().item()
    return {"accuracy": accuracy}

def setup_trainer(model, train_dataset, eval_dataset):
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./wav2vec2_classification",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=10,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    return trainer