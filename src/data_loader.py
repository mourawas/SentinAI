import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizerFast
from collections import Counter
import numpy as np
import random
from tqdm import tqdm
from datasets import load_dataset

    
class HatexplainDataset:
    def __init__(self, bert_model_name="bert-base-uncased", mask_probability=0.5):
        self.mask_probability = mask_probability
        self.dataset = None
        self.label_mapping = {0: "hate", 1: "normal", 2: "offensive"}
        
        # Load the BERT tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
        
        # Process the
        self.processed_dataset = self.process_dataset()
        
        print("Initialized Hatexplain Dataset")

        
    def _average_rationales(self, rationales, length):
        # Compute binary average of token rationales across annotators
        wordtoken_rationales = []
        for i in range(length):
            annotator_values = [r[i] for r in rationales if i < len(r)]
            if annotator_values:
                mean_value = np.mean(annotator_values)
                wordtoken_rationales.append(1 if mean_value >= 0.5 else 0)
            else:
                wordtoken_rationales.append(0)
        return wordtoken_rationales
    
    def _process_example(self, example):
        # Process a single data example: extract text, label, rationale and apply masking
        wordtext_tokens = example["post_tokens"]
        labels = example["annotators"]["label"]
        rationales = example["rationales"]
        label_counts = Counter(labels)
        most_common = label_counts.most_common()
        
        # Skip examples with no majority label
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            return None

        majority_label = most_common[0][0]
    
        if majority_label == 1:
            # If "normal", set rationale to all 0s
            rationale = [0] * len(wordtext_tokens)
        else: 
            # Use rationales only if valid
            valid_rationales = [r for r in rationales if any(r)]
            if not valid_rationales:
                return None
            rationale = self._average_rationales(rationales, len(wordtext_tokens))

        # Mask rationale tokens probabilistically
        masked_rationale = []
        mask_positions = []
        for r in rationale:
            if random.random() < self.mask_probability:
                masked_rationale.append(0)
                mask_positions.append(1)
            else:
                masked_rationale.append(r)
                mask_positions.append(0)
        
        # Tokenize with BERT
        encoder = self.tokenizer(wordtext_tokens, padding='max_length', is_split_into_words=True,
                                 truncation=True, max_length=128, return_tensors='pt')
        
        # Get the word_ids (maps each token to a word index)
        word_ids = encoder.word_ids(batch_index=0)
        
        # Map rationales to BERT tokens
        bert_rationale = []
        bert_masked_rationale = []
        bert_mask_positions = []
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens ([CLS], [SEP], padding)
                bert_rationale.append(0)
                bert_masked_rationale.append(0)
                bert_mask_positions.append(0)
            else:
                bert_rationale.append(rationale[word_idx])
                bert_masked_rationale.append(masked_rationale[word_idx])
                bert_mask_positions.append(mask_positions[word_idx])
         
        # Truncate if necessary
        bert_rationale = bert_rationale[:128]
        bert_masked_rationale = bert_masked_rationale[:128]
        bert_mask_positions = bert_mask_positions[:128]
        
        # Convert into tensor format
        bert_rationale = torch.tensor([bert_rationale], dtype=torch.long)
        bert_masked_rationale = torch.tensor([bert_masked_rationale], dtype=torch.long)
        bert_mask_positions = torch.tensor([bert_mask_positions], dtype=torch.long)
        
        text = ' '.join(wordtext_tokens)
        
        return text, majority_label, rationale, masked_rationale, mask_positions, encoder["input_ids"][0], encoder["attention_mask"][0], encoder["token_type_ids"][0], bert_rationale, bert_masked_rationale, bert_mask_positions
    
    def load_data(self):
        # Load HateXplain dataset from Hugging Face
        print("Loading HateXplain dataset...")
        self.dataset = load_dataset("hatexplain")
    
    def process_dataset(self):
        # Process the full dataset split-by-split
        if self.dataset is None:
            self.load_data()
        
        processed_dataset = {}
        for split in self.dataset:
            print(f"Processing {split} split...")
            examples = self.dataset[split]
            
            processed_examples = {
                "text": [],
                "label": [],
                "rationale": [],
                "masked_rationale": [],
                "mask_positions": [],
                "input_ids": [],
                "attention_mask": [],
                "token_type_ids": [],
                "bert_rationale": [],
                "bert_masked_rationale": [],
                "bert_mask_positions": []
                
            }
            
            skipped_count = 0
            
            for example in tqdm(examples):
                result = self._process_example(example)
                if result is None:
                    skipped_count += 1
                    continue
            
                text, label, rationale, masked_rationale, mask_positions, input_ids, attention_mask, token_type_ids, bert_rationale, bert_masked_rationale, bert_mask_positions = result
            
                processed_examples["text"].append(text)
                processed_examples["label"].append(torch.tensor(label, dtype=torch.long))
                processed_examples["rationale"].append(rationale)
                processed_examples["masked_rationale"].append(masked_rationale)
                processed_examples["mask_positions"].append(mask_positions)
                processed_examples["input_ids"].append(input_ids)
                processed_examples["attention_mask"].append(attention_mask)
                processed_examples["token_type_ids"].append(token_type_ids)
                processed_examples["bert_rationale"].append(bert_rationale)
                processed_examples["bert_masked_rationale"].append(bert_masked_rationale)
                processed_examples["bert_mask_positions"].append(bert_mask_positions)

            print(f"Skipped {skipped_count} examples")
            processed_dataset[split] = processed_examples
        
        return processed_dataset
    
    def get_split(self, split_name):
        # Retrieve a processed split ("train", "validation", or "test")
        if split_name not in self.processed_dataset:
            raise ValueError(f"Split '{split_name}' not found, possible split names are {self.processed_dataset.keys()}")
        return self.processed_dataset[split_name]

    def example_display(self, split_name, num_examples=3):
        print("\nExample processed rationales:")

        data = self.get_split(split_name)
        
        # Color Codes
        MAKSED_COLOR = "\033[94m" # Blue
        RATIONALE_COLOR = "\033[32m"  # Green
        MASKED_RATIONALE_COLOR = "\033[31m"     # Red
        RESET_COLOR = "\033[0m"       # Reset to normal
        
        # Display examples from each class
        for target_class in [0, 1, 2]:  # hate, normal, offensive
            examples_indices = [i for i, label in enumerate(data["label"]) if label == target_class]

            # Randomly sample `num_examples` examples from the selected class
            selected_indices = random.sample(examples_indices, min(num_examples, len(examples_indices)))
            for idx in selected_indices:
                label_name = self.label_mapping[target_class]
                print("\nLabel: ", label_name)
                
                # Display text with rationale highlighted
                text = data["text"][idx].split()
                rationale = data["rationale"][idx]
                masked_rationale = data["masked_rationale"][idx]
                mask_positions = data["mask_positions"][idx]
                
                print("Text with rationale (green), masked rationale (red), and masked positions (blue):")
                for token, is_rationale, is_masked, is_mask_position in zip(text, rationale, masked_rationale, mask_positions):
                    if is_mask_position == 1 and is_rationale == 0:
                        # Highlight masked positions in blue
                        print(f"{MAKSED_COLOR}{token}{RESET_COLOR}", end=" ")
                    elif is_masked == 1 and is_rationale == 1:
                        # Highlight rationale words in green
                        print(f"{RATIONALE_COLOR}{token}{RESET_COLOR}", end=" ")
                    elif is_mask_position == 1 and is_rationale == 1:
                        # Highlight masked rationale words in red
                        print(f"{MASKED_RATIONALE_COLOR}{token}{RESET_COLOR}", end=" ")
                    else:
                        print(token, end=" ")
                print()
                
                # print("\nTokenized BERT input:")
                # input_ids = data["input_ids"][idx]
                # bert_rationale = data["bert_rationale"][idx]
                # bert_masked_rationale = data["bert_masked_rationale"][idx]
                # bert_mask_positions = data["bert_mask_positions"][idx]
                
                # # Decode tokens and align rationale info
                # tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
                # for tok, r, mr, mp in zip(tokens, bert_rationale.squeeze(0), bert_masked_rationale.squeeze(0), bert_mask_positions.squeeze(0)):
                #     if mp == 1 and r == 0:
                #         # Highlight masked positions in blue
                #         print(f"{MAKSED_COLOR}[94m{tok}{RESET_COLOR}", end=" ")
                #     elif mr == 1 and r == 1:
                #         # Rationale token (green)
                #         print(f"{RATIONALE_COLOR}{tok}{RESET_COLOR}", end=" ")
                #     elif mp == 1 and r == 1:
                #         # Masked rationale (red)
                #         print(f"{MASKED_RATIONALE_COLOR}{tok}{RESET_COLOR}", end=" ")
                #     else:
                #         print(tok, end=" ")
                # print()
                
# Create data loaders for training
def create_data_loaders(dataset, split_name, batch_size=16):
    data = dataset.get_split(split_name)
    
    # Stack all tensors
    input_ids = torch.stack(data["input_ids"])
    attention_mask = torch.stack(data["attention_mask"])
    bert_rationale = torch.stack(data["bert_rationale"])
    bert_masked_rationale = torch.stack(data["bert_masked_rationale"])
    bert_mask_positions = torch.stack(data["bert_mask_positions"])
    labels = torch.stack(data["label"])
    
    # Create TensorDataset
    tensor_dataset = TensorDataset(input_ids, attention_mask, bert_rationale, bert_masked_rationale, bert_mask_positions, labels)
    
    # Create DataLoader
    data_loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=(split_name == "train"))
    
    return data_loader

