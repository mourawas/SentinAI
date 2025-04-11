from transformers import AutoModelForSequenceClassification

class SentinAI:
    def __init__(self, 
                 model_name="google/bert_uncased_L-4_H-128_A-2", 
                 num_labels=3
                 ):
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.id2label = {0: "HATESPEECH", 1: "NORMAL", 2: "OFFENSIVE"}
        self.label2id = {v: k for k, v in self.id2label.items()}

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )

    def get_model(self):
        return self.model

    def print_num_parameters(self):
        print(f"{self.model_name} number of parameters: {self.model.num_parameters()}")
