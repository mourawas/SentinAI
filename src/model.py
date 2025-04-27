import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertEncoder

class MRPModel(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased",  num_transformer_layers=2, dropout_prob=0.2):
        super(MRPModel, self).__init__()
        """
        Initialize the Masked Rationale Prediction (MRP) model
        """
        # Load pre-trained BERT model for text embeddings
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_config = self.bert.config
        self.hidden_size = self.bert_config.hidden_size
        
        # Create rationale embedding layer
        self.rationale_embeddings = nn.Embedding(num_embeddings=2, # Binary labels (0, 1)
                                                 embedding_dim=self.hidden_size) # Same dimension as BERT embeddings
        
        # Initialize the rationale embeddings with a small random values
        self.rationale_embeddings.weight.data.normal_(mean=0.0, std=0.02)
        
        # Use BertConfig to ensure compatibility with BERT's architecture
        transformer_config = BertConfig(hidden_size=self.hidden_size, 
                                        num_hidden_layers=num_transformer_layers,
                                        num_attention_heads=self.bert_config.num_attention_heads,
                                        intermediate_size=self.bert_config.intermediate_size,
                                        hidden_dropout_prob=dropout_prob,
                                        attention_probs_dropout_prob=dropout_prob
                                        )
        
        # Define the transformer layers using BertEncoder
        
        self.transformer_layers = BertEncoder(transformer_config)
        
        # Define MLP for final prediction
        self.rationale_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(self.hidden_size, 2) # Binary labels (0, 1)
        )
        
        # Print model initialization information
        print(f"Initialized MRP model")
    
    def forward(self, input_ids, attention_mask, bert_masked_rationale):
        """
        Forward pass of the MRP model.
        """
        
        # Get text embeddings from BERT
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                 return_dict=True)
        
        text_embeddings = bert_outputs.last_hidden_state # [batch_size, seq_len, hidden_size]
        
        # Get rationale embeddings and apply masking
        bert_masked_rationale = bert_masked_rationale.view(input_ids.size(0), -1)
        rationale_embeddings = self.rationale_embeddings(bert_masked_rationale) # [batch_size, seq_len, hidden_size]
        
        # Initial hidden state
        hidden_state = text_embeddings + rationale_embeddings
        
        # Process through transformer layers
        # Create proper attention mask for transformer
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Pass the hidden state and attention mask to the transformer
        transformer_outputs = self.transformer_layers(hidden_state,
                                                      attention_mask=extended_attention_mask,
                                                      return_dict=True)
        
        final_hidden_state = transformer_outputs.last_hidden_state
        
        # Apply MLP to get rationale predictions
        rationale_logits = self.rationale_predictor(final_hidden_state)
        
        return rationale_logits
    
    def compute_loss(self, logits, labels, masked_positions):
        """
        Compute the loss of the MRP model.
        """
        # Use cross-entropy loss
        loss_fn = nn.CrossEntropyLoss()
                
        # Flatten the tensors
        logits_flat = logits.view(-1, 2)  # [batch_size * seq_len, 2]
        labels_flat = labels.view(-1)  # [batch_size * seq_len]
        mask_flat = masked_positions.view(-1)  # [batch_size * seq_len]
        
        # Get only the positions that were masked
        masked_indices = torch.nonzero(mask_flat, as_tuple=True)[0]
        
        if len(masked_indices) == 0:
            # No masked positions, return zero loss
            return torch.tensor(0.0, device=logits_flat.device, requires_grad=True)
        
        # Select logits and labels only at masked positions
        masked_logits = logits_flat[masked_indices]  # [num_masked, 2]
        masked_labels = labels_flat[masked_indices]  # [num_masked]
        
        # Calculate loss only on masked positions
        loss = loss_fn(masked_logits, masked_labels)
        
        return loss
    
class SentinAI(nn.Module):
    def __init__(self, mrp_model, num_classes=3, dropout_prob=0.1):
        super(SentinAI, self).__init__()
        """
        Initialize the Hate Speech Detection model using parameters from a trained MRP model
        """
        # Copy the pre-trained BERT and transformer layers from the MRP model
        self.bert = mrp_model.bert  # Use the BERT model from MRP
        self.transformer_layers = mrp_model.transformer_layers  # Use the transformer layers from MRP
        self.hidden_size = mrp_model.hidden_size
        
        # Preserve the rationale embeddings layer
        self.rationale_embeddings = mrp_model.rationale_embeddings
        
        # Replace the rationale prediction head with a classification head for hate speech detection
        self.classification_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_prob),
            nn.Linear(self.hidden_size, num_classes)  # 3 classes: hate speech, offensive, normal
        )
        
        print("Initialized SentinAI model")
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the SentinAI model.
        """
        # Get text embeddings from Bert
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                 return_dict=True)
        text_embeddings = bert_outputs.last_hidden_state # [batch_size, seq_len, hidden_size]
        
        # Set rationale embeddings to zero vectors as mentioned in the paper
        batch_size, seq_len = input_ids.size()
        zero_rationale_embeddings = torch.zeros(batch_size, seq_len, self.hidden_size, device=input_ids.device)
        
        # Initial hidden state
        hidden_state = text_embeddings + zero_rationale_embeddings
        
        # Process through transformer layers
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Pass the hidden state and attention mask to the transformer
        transformer_outputs = self.transformer_layers(hidden_state,
                                                      attention_mask=extended_attention_mask,
                                                      return_dict=True)
        
                
        final_hidden_state = transformer_outputs.last_hidden_state
        
        # Use [CLS] token representation for classification
        cls_representation = final_hidden_state[:, 0, :]
        
        # Apply classification head to get predictions
        logits = self.classification_head(cls_representation)
        
        return logits