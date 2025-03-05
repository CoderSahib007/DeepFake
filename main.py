# main.py
import torch
from transformers import Wav2Vec2Processor
from datasets import Dataset
from model import Wav2Vec2ForAudioClassification
from preprocess import preprocess_audio
from train import setup_trainer
import pandas as pd
import torchaudio

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print("================================================= Loading Dataset =================================================")
    # Loading the dataset
    manifest_path = "D:\\DeepFakeClassification\\archive\\KAGGLE\\AUDIO\\COMBINED_MANIFEST.csv"
    df = pd.read_csv(manifest_path)

    # Construct the expected format
    data = {
        "audio": df["audio_path"].tolist(),  # Use absolute paths directly
        "labels": df["label"].tolist()       # Ensure "label" column exists in the manifest
    }

    dataset = Dataset.from_dict(data)
    print("============================================ Dataset Loaded Successfully =========================================")

    # Precompute global maximum length across all audio files
    max_length = 0
    for example in dataset:
        audio_path = example["audio"]
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:  # Match sampling rate used in preprocess_audio
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:  # Convert to mono
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        max_length = max(max_length, waveform.shape[1])
    print(f"Global max length computed: {max_length}")

    # Split dataset into train and test sets
    train_test_split = dataset.train_test_split(test_size=0.2)  # 80% train, 20% test
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    print("================================================= Loading Model ==================================================")
    # Load processor and model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForAudioClassification.from_pretrained("facebook/wav2vec2-base-960h")
    model.to(device)
    # model.freeze_base_model()
    print("=========================================== Model Loaded Successfully ============================================")

    print("========================================== Preprocessing Dataset =================================================")
    # Preprocess dataset with perturbations
    def preprocess_batch(examples):
        processed = preprocess_audio(examples["audio"], processor, model=model, max_length=max_length)
        processed["labels"] = examples["labels"]
        return processed

    # Apply preprocessing to train and test datasets
    processed_train_dataset = train_dataset.map(preprocess_batch, batched=True, batch_size=2)
    processed_test_dataset = test_dataset.map(preprocess_batch, batched=True, batch_size=2)

    print("=================================== Dataset Preprocessed Successfully  ==========================================")

    print("============================================ Training Started  ==================================================")
    # Setup and train (use test dataset as eval dataset)
    trainer = setup_trainer(model, processed_train_dataset, processed_test_dataset)
    trainer.train()

    print("============================================ Training Completed  ================================================")

    # Inference example
    test_audio = ["D:\\DeepFakeClassification\\archive\\KAGGLE\\AUDIO\\test_audio.wav"]  # Replace with actual test audio path
    processed_test = preprocess_audio(test_audio, processor, model=None, max_length=max_length)  # No perturbation during inference
    input_values = processed_test["input_values"].to(device)
    attention_mask = processed_test["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_values, attention_mask=attention_mask)
        logits = outputs.logits
        prediction = (torch.sigmoid(logits) > 0.5).int().item()
        print(f"Predicted class: {prediction}")

if __name__ == "__main__":
    main()