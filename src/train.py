import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from utils import save_checkpoint
from sklearn.metrics import accuracy_score, f1_score

# Define evaluation metrics
def calculate_metrics(logits, labels, masked_positions):
    
    # Flatten the tensors
    logits_flat = logits.view(-1, 2)  # [batch_size * seq_len, 2]
    labels_flat = labels.view(-1)  # [batch_size * seq_len]
    mask_flat = masked_positions.view(-1)  # [batch_size * seq_len]
    
    
    # Convert logits to predictions (0 or 1)
    preds_flat = logits_flat.argmax(dim=-1)
    
    # Get only the positions that were masked
    masked_indices = torch.nonzero(mask_flat, as_tuple=True)[0]

    if len(masked_indices) == 0:
        # No masked positions, return zero loss
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # Only consider masked positions
    masked_preds = preds_flat[masked_indices]
    masked_labels = labels_flat[masked_indices]
        
    # Calculate accuracy
    correct = (masked_preds == masked_labels).sum().item()
    total = masked_positions.sum().item()    
    accuracy = correct / total
    
    # Calculate precision, recall, F1 score for the positive class (1)
    true_positives = ((masked_preds == 1) & (masked_labels == 1)).sum().item()
    false_positives = ((masked_preds == 1) & (masked_labels == 0)).sum().item()
    false_negatives = ((masked_preds == 0) & (masked_labels == 1)).sum().item()
    
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
# Training function
def train_mrp_model(model, optimizer, train_loader, val_loader, num_epochs=3, start_epoch=0, device="cuda", result_path="../outputs", model_path="../model"):
    # Check if GPU is available
    if torch.cuda.is_available() and device == "cuda":
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU")
        
    model = model.to(device)
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": []
    }
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Training")
        for batch in progress_bar:
            # Move batch to device
            batch = [b.to(device) for b in batch]
            input_ids, attention_mask, bert_rationale, bert_masked_rationale, bert_mask_positions, _ = batch
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            rationale_logits = model(input_ids, attention_mask, bert_masked_rationale)
            
            # Compute loss
            loss = model.compute_loss(rationale_logits, bert_rationale, bert_mask_positions)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Update training loss
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        # Average training loss for the epoch
        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                batch = [b.to(device) for b in batch]
                input_ids, attention_mask, bert_rationale, bert_masked_rationale, bert_mask_positions, _ = batch
                
                # Forward pass
                rationale_logits = model(input_ids, attention_mask, bert_masked_rationale)
                
                # Compute loss
                loss = model.compute_loss(rationale_logits, bert_rationale, bert_mask_positions)
                val_loss += loss.item()
                
                # Calculate metrics
                batch_metrics = calculate_metrics(rationale_logits, bert_rationale, bert_mask_positions)
                all_metrics.append(batch_metrics)
        
        # Average validation loss and metrics
        avg_val_loss = val_loss / len(val_loader)
        history["val_loss"].append(avg_val_loss)
        
        # Aggregate metrics across batches
        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics if m[key] is not None]) 
            for key in ["accuracy", "precision", "recall", "f1"]
        }
        
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch+1, avg_train_loss, 
                        result_path/"checkpoints/mrp_checkpoint.pth", 
                        store_checkpoint_for_every_epoch=False)
        
        history["val_accuracy"].append(avg_metrics["accuracy"])
        history["val_f1"].append(avg_metrics["f1"])
        
        # Print epoch results
        print(f"Train Loss: {avg_train_loss:4f}")
        print(f"Val Loss: {avg_val_loss:4f}")
        print(f"Val Metrics: {avg_metrics}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path/"best_mrp_model.pt")
            print(f"Saved best MRP model at Epoch {epoch+1}/{num_epochs}!")
    
    return history

# Training function
def train_sentinai_model(model, optimizer, train_loader, val_loader, num_epochs=3, start_epoch=0, device="cuda", result_path="../outputs", model_path="../model"):
    # Check if GPU is available
    if torch.cuda.is_available() and device == "cuda":
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU")
        
    model = model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": []
    }
    best_val_loss = float('inf')
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Training")
        for batch in progress_bar:
            # Move batch to device
            batch = [b.to(device) for b in batch]
            input_ids, attention_mask, _, _, _, labels = batch
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            
            # Compute loss
            loss = loss_fn(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Update training loss
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        # Average training loss for the epoch
        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                batch = [b.to(device) for b in batch]
                input_ids, attention_mask, _, _, _, labels = batch
                
                # Forward pass
                logits = model(input_ids, attention_mask)
                
                # Compute loss
                loss = loss_fn(logits, labels)
                val_loss += loss.item()
                
                # Save predictions and labels for metrics calculation
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        # Average validation loss and metrics
        avg_val_loss = val_loss / len(val_loader)
        history["val_loss"].append(avg_val_loss)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        
                
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch+1, avg_train_loss, 
                        result_path/"checkpoints/sentinai_checkpoint.pth", 
                        store_checkpoint_for_every_epoch=False)
        
        history["val_accuracy"].append(accuracy)
        history["val_f1"].append(macro_f1)
        
        # Print epoch results
        print(f"Train Loss: {avg_train_loss:4f}")
        print(f"Val Loss: {avg_val_loss:4f}")
        print(f"Val Metrics: {accuracy:4}, {macro_f1:4}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path/"best_sentinai_model.pt")
            print(f"Saved best SentinAI model at Epoch {epoch+1}/{num_epochs}!")
    
    return history