from datasets import load_dataset, Dataset, DatasetDict
from collections import Counter
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

class HateXplainDataLoader:
    def __init__(self, 
                 model_name="google/bert_uncased_L-2_H-128_A-2", 
                 max_length=512, 
                 batch_size=64,
                 cache_dir="~/.cache/hatexplain"):
        
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_length)

    def process_dataset(self, raw_data):
        processed_splits = {}
        for split in ["train", "validation", "test"]:
            split_data = []
            for item in raw_data[split]:
                text = " ".join(item["post_tokens"])
                labels = item["annotators"]["label"]
                label = Counter(labels).most_common(1)[0][0]
                split_data.append({"text": text, "label": label})
            processed_splits[split] = Dataset.from_list(split_data)
        return DatasetDict(processed_splits)

    def tokenize_dataset(self, dataset):
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True)

        cache_files = {
            "train": f"{self.cache_dir}/hatexplain_train_tokenized.arrow",
            "validation": f"{self.cache_dir}/hatexplain_val_tokenized.arrow",
            "test": f"{self.cache_dir}/hatexplain_test_tokenized.arrow"
        }

        tokenized_dataset = dataset.map(tokenize_function, batched=True, cache_file_names=cache_files)
        tokenized_dataset = tokenized_dataset.remove_columns(["text"])
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        tokenized_dataset.set_format("torch")
        return tokenized_dataset

    def get_dataloaders(self):
        raw_data = load_dataset("hatexplain")
        processed_data = self.process_dataset(raw_data)
        tokenized_data = self.tokenize_dataset(processed_data)

        train_loader = DataLoader(tokenized_data["train"], shuffle=True, batch_size=self.batch_size)
        val_loader = DataLoader(tokenized_data["validation"], shuffle=True, batch_size=self.batch_size)
        test_loader = DataLoader(tokenized_data["test"], batch_size=self.batch_size)

        return train_loader, val_loader, test_loader

