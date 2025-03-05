import torch
import torch.nn as nn
from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Model
from transformers.modeling_outputs import SequenceClassifierOutput

class Wav2Vec2ForAudioClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        # Base wav2vec2 model
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)
        
        # Classification head: Linear layer for binary output (0 or 1)
        hidden_size = config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),  # Reduce dimensionality
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)  # Single output for binary classification
        )
        
        # Initialize weights
        self.post_init()
    
    def freeze_feature_encoder(self):
        """Freeze the feature encoder to prevent updates during training."""
        self.wav2vec2.feature_extractor._freeze_parameters()
    
    def freeze_base_model(self):
        """Freeze the base model, only train the classifier."""
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Extract features from wav2vec2
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Use the last hidden state (CLS-like pooling: mean over time)
        hidden_states = outputs[0]  # Shape: (batch_size, sequence_length, hidden_size)
        pooled_output = hidden_states.mean(dim=1)  # Mean pooling: (batch_size, hidden_size)
        pooled_output = self.dropout(pooled_output)
        
        # Classification logits
        logits = self.classifier(pooled_output)  # Shape: (batch_size, 1)
        
        # Loss computation
        loss = None
        if labels is not None:
            # Ensure labels are float for BCEWithLogitsLoss
            labels = labels.view(-1, 1).float()
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )