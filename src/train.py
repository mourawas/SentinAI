import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import evaluate

from data_loader import HateXplainDataLoader
from model import SentinAI
from utils import save_checkpoint, load_checkpoint

def parse_args():
    #when working with python files from console it's better to specify
    parser = argparse.ArgumentParser(description="File creation script.")
    #parser.add_argument("--dataset_path", required=True, help="Dataset path")
    parser.add_argument("--results_path", required=True, help="Output directory")

    args = parser.parse_args()

    #return args.dataset_path, args.results_path
    return args.results_path

# DATADIR, RESULTS_DIR = parse_args()
RESULTS_DIR = parse_args()
if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)
# Directory to store model checkpoints to
if not os.path.exists(Path(RESULTS_DIR) / 'checkpoints'):
    os.mkdir(Path(RESULTS_DIR) / 'checkpoints')
    
class HateXplainTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, metrics, device, lr=5e-5, num_epochs=1, start_epoch=0, warmup_steps=1):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.metrics = metrics
        self.device = device
        self.lr = lr
        self.num_epochs = num_epochs
        self.start_epoch = start_epoch
        self.warmup_steps = warmup_steps

        self.num_training_steps = num_epochs * len(train_dataloader)

        self.lr_scheduler = get_scheduler(name="linear", optimizer=self.optimizer,
                                          num_warmup_steps=self.warmup_steps,
                                          num_training_steps=self.num_training_steps
                                          )
        
        self.train_loss_log = []
        self.val_loss_log = []
        self.train_metrics_log = {k: [] for k in self.metrics.keys()}
        self.val_metrics_log = {k: [] for k in self.metrics.keys()}
    
    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        epoch_metrics = {k: 0 for k in self.metrics.keys()}
        
        progress_bar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))
        
        for batch_num, batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            
            # Compute predictions
            with torch.no_grad():
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
            
                for k, metric in self.metrics.items():
                    metric_value = metric.compute(predictions=predictions.cpu(), 
                                                 references=batch["labels"].cpu())
                    # Extract the value from the metric result
                    metric_value = list(metric_value.values())[0] if isinstance(metric_value, dict) else metric_value
                    epoch_metrics[k] += metric_value
            
            # Log loss
            epoch_loss += loss.item()
        
        # Average the loss and metrics over batches
        epoch_loss /= len(self.train_dataloader)
        for k in epoch_metrics.keys():
            epoch_metrics[k] /= len(self.train_dataloader)
        
        # Print metrics
        metrics_str = ', '.join([f'{k}: {v:.4f}' for k, v in epoch_metrics.items()])
        print(f'Train Loss: {epoch_loss:.4f}, {metrics_str}')
        
        return epoch_loss, epoch_metrics
    
    def evaluate(self):
        self.model.eval()
        epoch_loss = 0
        epoch_metrics = {k: 0 for k in self.metrics.keys()}
        
        progress_bar = tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader))
        
        for batch_num, batch in progress_bar:
            
            with torch.no_grad():
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Compute predictions
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                
                # Compute metrics
                for k, metric in self.metrics.items():
                    metric_value = metric.compute(predictions=predictions.cpu(), 
                                                references=batch["labels"].cpu())
                    # Extract the value from the metric result
                    metric_value = list(metric_value.values())[0] if isinstance(metric_value, dict) else metric_value
                    epoch_metrics[k] += metric_value
                
                # Log loss
                epoch_loss += loss.item()
            
            # Average the loss and metrics over batches
            epoch_loss /= len(self.val_dataloader)
            for k in epoch_metrics.keys():
                epoch_metrics[k] /= len(self.val_dataloader)
            
        # Print metrics
        metrics_str = ', '.join([f'{k}: {v:.4f}' for k, v in epoch_metrics.items()])
        print(f'Eval Loss: {epoch_loss:.4f}, {metrics_str}')
            
        return epoch_loss, epoch_metrics
                
    def train(self):
        
        for epoch in range(self.start_epoch, self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Train for one epoch
            train_loss, train_metrics = self.train_epoch()
            self.train_loss_log.append(train_loss)
            self.update_metrics_log(train_metrics, is_train=True)
            
            # Evaluate
            val_loss, val_metrics = self.evaluate()
            self.val_loss_log.append(val_loss)
            self.update_metrics_log(val_metrics, is_train=False)
            
            # Plot training progress
            self.plot_training()
            
            # Save checkpoint
            save_checkpoint(self.model, self.optimizer, epoch, train_loss, 
                            str(Path(RESULTS_DIR) / "checkpoints/checkpoint.pth"),
                            store_checkpoint_for_every_epoch=(epoch + 1) % 5 == 0 or (epoch + 1) == self.num_epochs
            )
            
    def plot_training(self):
        metrics_names = list(self.metrics.keys())
        fig, ax = plt.subplots(1, len(metrics_names) + 1, figsize=((len(metrics_names) + 1) * 5, 5))
        
        # Plot loss
        ax[0].plot(self.train_loss_log, c='blue', label='train')
        ax[0].plot(self.val_loss_log, c='orange', label='validation')
        ax[0].set_title('Loss')
        ax[0].set_xlabel('epoch')
        ax[0].legend()
        
        # Plot metrics
        for i, metric_name in enumerate(metrics_names):
            ax[i + 1].plot(self.train_metrics_log[metric_name], c='blue', label='train')
            ax[i + 1].plot(self.val_metrics_log[metric_name], c='orange', label='validation')
            ax[i + 1].set_title(metric_name)
            ax[i + 1].set_xlabel('epoch')
            ax[i + 1].legend()
        
        plt.tight_layout()
        plt.savefig(Path(RESULTS_DIR) / "training_loss_and_metrics.jpg")
        plt.close()

    def update_metrics_log(self, metrics_dict, is_train=True):
        target_log = self.train_metrics_log if is_train else self.val_metrics_log
        for metric_name, value in metrics_dict.items():
            target_log[metric_name].append(value)
            
def main():
    # Load data
    data_loader = HateXplainDataLoader()
    train_loader, val_loader, _ = data_loader.get_dataloaders()

    # Load model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_loader = SentinAI()
    model = model_loader.get_model()
    model.to(device)
    print(f"Using device: {device}")

    # Parameters
    lr = 5e-5
    num_epochs = 100
    metrics = {
        "accuracy": evaluate.load("accuracy"),
    } 
    warmup_steps = 1
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr)
    
    saved_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path = Path(RESULTS_DIR) / "checkpoints/checkpoint.pth")
    if saved_epoch == 0:
        start_epoch = 0
    else:
        start_epoch = saved_epoch + 1  #if the checkpoint from the epoch saved_epoch is stored, we want to start the training from the next epoch

    # Train
   
    trainer = HateXplainTrainer(model, train_loader, val_loader, 
                                optimizer, metrics, device, lr=lr, 
                                num_epochs=num_epochs, start_epoch=start_epoch, 
                                warmup_steps=warmup_steps)
    trainer.train()

if __name__ == "__main__":
    main()
