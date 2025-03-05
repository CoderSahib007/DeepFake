# preprocess.py
import torch
import torchaudio
from transformers import Wav2Vec2Processor

def preprocess_audio(audio_paths, processor, model=None, sampling_rate=16000, perturbation_steps=3, epsilon=0.01, max_length=None):
    """Preprocess audio files with F-SAT perturbations for robustness training."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_values = []
    
    # If max_length is not provided, compute it from the current batch
    if max_length is None:
        max_length = 0
        for audio_path in audio_paths:
            waveform, sr = torchaudio.load(audio_path)
            if sr != sampling_rate:
                resampler = torchaudio.transforms.Resample(sr, sampling_rate)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:  # Convert to mono
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            audio_length = waveform.shape[1]
            max_length = max(max_length, audio_length)
    
    # Process and pad audio
    for audio_path in audio_paths:
        # Load and resample audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, sampling_rate)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:  # Convert to mono
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Convert to tensor and ensure 1D
        audio = waveform.squeeze().to(device).float()
        
        # Apply F-SAT perturbations if model is provided (for training)
        if model is not None and model.training:
            audio = audio.unsqueeze(0)  # Add batch dimension
            
            # STFT to get magnitude and phase
            window_size = 512
            hop_length = 128
            stft = torch.stft(audio, n_fft=window_size, hop_length=hop_length, return_complex=True)
            magnitude, phase = torch.abs(stft), torch.angle(stft)
            
            # Initialize perturbation
            perturbation = torch.zeros_like(magnitude, device=device, requires_grad=True)
            
            # Iterative perturbation (simplified F-SAT)
            for _ in range(perturbation_steps):
                # Reconstruct perturbed audio
                perturbed_magnitude = magnitude + perturbation
                perturbed_stft = perturbed_magnitude * torch.exp(1j * phase)
                perturbed_audio = torch.istft(perturbed_stft, n_fft=window_size, hop_length=hop_length, length=audio.shape[-1])
                
                # Process through model to get loss
                processed = processor(perturbed_audio.squeeze().cpu().numpy(), 
                                    sampling_rate=sampling_rate, 
                                    return_tensors="pt", padding=True)
                processed = {k: v.to(device) for k, v in processed.items()}
                
                outputs = model(processed["input_values"], attention_mask=processed["attention_mask"])
                loss = outputs.loss
                loss.backward()
                
                # Update perturbation (gradient ascent to maximize loss)
                with torch.no_grad():
                    perturbation += epsilon * perturbation.grad / (perturbation.grad.norm() + 1e-8)
                    perturbation.clamp_(-epsilon, epsilon)  # Constrain perturbation magnitude
                    perturbation.grad.zero_()
            
            # Apply final perturbation and reconstruct
            perturbed_magnitude = magnitude + perturbation
            perturbed_stft = perturbed_magnitude * torch.exp(1j * phase)
            perturbed_audio = torch.istft(perturbed_stft, n_fft=window_size, hop_length=hop_length, length=audio.shape[-1])
            audio = perturbed_audio.squeeze()
        
        # Pad audio to max_length
        if audio.shape[0] < max_length:
            padding = max_length - audio.shape[0]
            audio = torch.nn.functional.pad(audio, (0, padding))
        
        # Process with wav2vec2 processor
        audio_array = audio.cpu().numpy()
        inputs = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        input_values.append(inputs["input_values"].squeeze(0))
    
    # Pad sequences to the same length (should already be handled by max_length padding)
    input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)
    attention_mask = (input_values != 0).long()  # Mask for padded regions
    return {"input_values": input_values, "attention_mask": attention_mask}