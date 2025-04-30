import argparse
import os
from pathlib import Path
import torch
from torch.optim import RAdam
import numpy as np
import random

from data_loader import HatexplainDataset, create_data_loaders
from model import MRPModel
from train import train_mrp_model
from utils import load_checkpoint, plot_training_history

def parse_args():
    
    parser = argparse.ArgumentParser(description="File creation script.")
    #parser.add_argument("--dataset_path", required=True, help="Dataset path")
    parser.add_argument("--model_path", required=True, help="Model directory")
    parser.add_argument("--results_path", required=True, help="Output directory")

    args = parser.parse_args()

    #return args.dataset_path, args.results_path
    return args.model_path, args.results_path

MODEL_DIR, RESULTS_DIR = parse_args()
if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)
    
# Directory to store model checkpoints to
if not os.path.exists(Path(RESULTS_DIR) / 'checkpoints'):
    os.mkdir(Path(RESULTS_DIR) / 'checkpoints')
    
# Directory to store model logs to   
if not os.path.exists(Path(RESULTS_DIR) / 'logs'):
    os.mkdir(Path(RESULTS_DIR) / 'logs')
    
# Directory to store model parameters to  
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

def main():
    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Hyperparameters
    bert_model_name = "bert-base-uncased"
    mask_probability = 0.5
    batch_size = 16
    learning_rate = 5e-5
    num_epochs = 3
    num_transformer_layers = 6
    dropout_prob = 0.2

    # Create dataset
    dataset = HatexplainDataset(bert_model_name=bert_model_name, mask_probability=mask_probability)

    # Display some examples
    dataset.example_display("train", num_examples=1)
    
    # Create data loaders
    train_loader = create_data_loaders(dataset, "train", batch_size=batch_size)
    val_loader = create_data_loaders(dataset, "validation", batch_size=batch_size)
    
    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    mrp_model = MRPModel(bert_model_name=bert_model_name,
                         num_transformer_layers=num_transformer_layers,
                         dropout_prob=dropout_prob)
    mrp_model.to(device)
    
    # Initialize RAdam optimizer
    optimizer = RAdam(mrp_model.parameters(), lr=learning_rate)
    
    saved_epoch, _ = load_checkpoint(mrp_model, optimizer, checkpoint_path = Path(RESULTS_DIR)/"checkpoints/mrp_checkpoint.pth")
    if saved_epoch == 0:
        start_epoch = 0
    else:
        start_epoch = saved_epoch + 1  #if the checkpoint from the epoch saved_epoch is stored, we want to start the training from the next epoch
        

    history = train_mrp_model(model=mrp_model, optimizer=optimizer, train_loader=train_loader,
                              val_loader=val_loader, num_epochs=num_epochs, 
                              start_epoch=start_epoch, device=device,
                              result_path=Path(RESULTS_DIR), model_path=Path(MODEL_DIR))

    # Plot training history
    plot_training_history(history, result_path=Path(RESULTS_DIR), model="mrp")
       
if __name__ == "__main__":
    main()