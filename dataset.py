from torch.utils.data import Dataset

from transformers import BertTokenizer
import copy
import os
import json
import numpy as np
import emoji
import sys
import argparse

from utils import add_tokens_to_tokenizer, get_token_rationale

class HateXplainDataset(Dataset):
    def __init__(self, args, mode='train'):
        assert mode in ['train', 'val', 'test'], "the mode should be [train/val/test]"
        data_root = args.dir_hatexplain
        data_dir = os.path.join(data_root, 'hatexplain_thr_div.json')
        with open(data_dir, 'r') as f:
            dataset = json.load(f)

        self.label_list = ['hatespeech', 'normal', 'offensive']
        self.label_count = [0, 0, 0]
            
        if mode == 'train':
            self.dataset = dataset['train']
            for d in self.dataset:
                for i in range(len(self.label_list)):
                    if d['final_label'] == self.label_list[i]:
                        self.label_count[i] += 1         
        elif mode == 'val':
            self.dataset = dataset['val']
        else:  # 'test'
            self.dataset = dataset['test']
            
        if args.intermediate:  
            rm_idxs = []
            for idx, d in enumerate(self.dataset):
                if 1 not in d['final_rationale'] and d['final_label'] in ('offensive', 'hatespeech'):
                    rm_idxs.append(idx)
            rm_idxs.sort(reverse=True)
            for j in rm_idxs:
                del self.dataset[j]

        self.mode = mode
        self.intermediate = args.intermediate

        tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
        self.tokenizer = add_tokens_to_tokenizer(args, tokenizer)
        
        #### ADDED PARTS ####
        
        self.multitask = args.multitask and not args.intermediate
        
        if self.multitask:
            self.target_groups = [
                # Race
                'African', 'Arab', 'Asian', 'Caucasian', 'Hispanic', 
                # Religion
                'Buddhism', 'Christian', 'Hindu', 'Islam', 'Jewish',
                # Gender
                'Men', 'Women',
                # Sexual Orientation
                'Heterosexual', 'Homosexual',
                # Miscellaneous
                'Indigenous', 'Refugee', 'Other'
            ]
            
            self.num_target_groups = len(self.target_groups)
            
            # Print target stats only for training in multi-task mode
            # if mode == 'train':
            #     self._print_target_stats()
                
        #### END OF ADDED PARTS ####
        
    #### ADDED PARTS ####
        
    def _print_target_stats(self):
        """Print statistics about target group distribution"""
        target_counts = {group: 0 for group in self.target_groups}
        target_counts['None'] = 0
        
        multi_target_count = 0
        
        for item in self.dataset:
            targets = item.get('final_target_category', 'None')
            
            if isinstance(targets, list):
                if len(targets) > 1:
                    multi_target_count += 1
                
                for target in targets:
                    if target in target_counts:
                        target_counts[target] += 1
                    else:
                        print(f"Warning: Unknown target category '{target}'")
            else: 
                target_counts['None'] += 1
        
        print("\nTarget group distribution in training data:")
        for group, count in sorted(target_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {group}: {count} posts")
        print(f"Posts with multiple targets: {multi_target_count}")
        print(f"Total posts: {len(self.dataset)}\n")
        
    def _encode_target_groups(self, target_category):
        """Convert target category to multi-hot encoding - only used in multi-task mode"""
        
        if not self.multitask:
            raise ValueError("Multi-task encoding is only available in multi-task mode.")
        
        # Initialize with all zeros
        target_encoding = [0] * self.num_target_groups
        
        # Handle None case
        if target_category == 'None' or target_category is None:
            return target_encoding
        
        # Handle list case
        if isinstance(target_category, list):
            for target in target_category:
                if target != 'None':
                    try:
                        idx = self.target_groups.index(target)
                        target_encoding[idx] = 1
                    except ValueError:
                        print(f"Warning: Unknown target category '{target}'")
                        
        return target_encoding
    
    #### END OF ADDED PARTS ####

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        id = self.dataset[idx]['post_id']
        text = ' '.join(self.dataset[idx]['text'])
        text = emoji.demojize(text, use_aliases=True)
        label = self.dataset[idx]['final_label']
        cls_num = self.label_list.index(label)

        if self.intermediate:
            fin_rat = self.dataset[idx]['final_rationale']
            fin_rat_token = get_token_rationale(self.tokenizer, copy.deepcopy(text.split(' ')), copy.deepcopy(fin_rat), copy.deepcopy(id))

            tmp = []
            for f in fin_rat_token:
                tmp.append(str(f))
            fin_rat_str = ','.join(tmp)
            return (text, cls_num, fin_rat_str)
            
        elif self.intermediate == False:  # hate speech detection
            
            #### ADDED PARTS ####
            
            # Get target groups only if multi-task is enabled
            if self.multitask:
                target_category = self.dataset[idx].get('final_target_category', 'None')
                target_encoding = self._encode_target_groups(target_category)
                return (text, cls_num, id, target_encoding)
            else:
                return (text, cls_num, id)
            
            #### END OF ADDED PARTS ####
            
        else:  
            return ()

            
class HateXplainDatasetForBias(Dataset):
    def __init__(self, args, mode='train'):
        assert mode in ['train', 'val', 'test'], "mode should be [train/val/test]"
        
        data_root = args.dir_hatexplain
        data_dir = os.path.join(data_root, 'hatexplain_two_div.json')
        with open(data_dir, 'r') as f:
            dataset = json.load(f)

        self.label_list = ['non-toxic', 'toxic']
        self.label_count = [0, 0]

        if mode == 'train':
            self.dataset = dataset['train']
            for d in self.dataset:
                if d['final_label'] == self.label_list[0]:
                    self.label_count[0] += 1
                elif d['final_label'] == self.label_list[1]:
                    self.label_count[1] += 1
                else:
                    print("[!] exceptional label ", d['final_label'])
                    return
        elif mode == 'val':
            self.dataset = dataset['val']
        else:  # 'test'
            self.dataset = dataset['test']
        self.mode = mode
        self.intermediate = args.intermediate
        assert self.intermediate == False
        
        #### ADDED PARTS ####
        
        self.multitask = args.multitask and not args.intermediate
        
        if self.multitask:
            self.target_groups = [
                # Race
                'African', 'Arab', 'Asian', 'Caucasian', 'Hispanic', 
                # Religion
                'Buddhism', 'Christian', 'Hindu', 'Islam', 'Jewish',
                # Gender
                'Men', 'Women',
                # Sexual Orientation
                'Heterosexual', 'Homosexual',
                # Miscellaneous
                'Indigenous', 'Refugee', 'Other'
            ]
            
            self.num_target_groups = len(self.target_groups)
            
            # Print target stats only for training in multi-task mode
            # if mode == 'train':
            #     self._print_target_stats()
                
        #### END OF ADDED PARTS ####
        
    #### ADDED PARTS ####
        
    def _print_target_stats(self):
        """Print statistics about target group distribution"""
        target_counts = {group: 0 for group in self.target_groups}
        target_counts['None'] = 0
        
        multi_target_count = 0
        
        for item in self.dataset:
            targets = item.get('final_target_category', 'None')
            
            if isinstance(targets, list):
                if len(targets) > 1:
                    multi_target_count += 1
                
                for target in targets:
                    if target in target_counts:
                        target_counts[target] += 1
                    else:
                        print(f"Warning: Unknown target category '{target}'")
            else: 
                target_counts['None'] += 1
        
        print("\nTarget group distribution in training data:")
        for group, count in sorted(target_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {group}: {count} posts")
        print(f"Posts with multiple targets: {multi_target_count}")
        print(f"Total posts: {len(self.dataset)}\n")
        
    def _encode_target_groups(self, target_category):
        """Convert target category to multi-hot encoding - only used in multi-task mode"""
        
        if not self.multitask:
            raise ValueError("Multi-task encoding is only available in multi-task mode.")
        
        # Initialize with all zeros
        target_encoding = [0] * self.num_target_groups
        
        # Handle None case
        if target_category == 'None' or target_category is None:
            return target_encoding
        
        # Handle list case
        if isinstance(target_category, list):
            for target in target_category:
                if target != 'None':
                    try:
                        idx = self.target_groups.index(target)
                        target_encoding[idx] = 1
                    except ValueError:
                        print(f"Warning: Unknown target category '{target}'")
                        
        return target_encoding
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        post_id = self.dataset[idx]['post_id']
        text = ' '.join(self.dataset[idx]['text'])
        text = emoji.demojize(text)
        label = self.dataset[idx]['final_label']
        cls_num = self.label_list.index(label)
        
        #### ADDED PARTS ####
            
        # Get target groups only if multi-task is enabled
        if self.multitask:
            target_category = self.dataset[idx].get('final_target_category', 'None')
            target_encoding = self._encode_target_groups(target_category)
            return (text, cls_num, post_id, target_encoding)
        else:
            return (text, cls_num, post_id)
        
        #### END OF ADDED PARTS ####

if __name__ == '__main__':
    def get_args_1():
        parser = argparse.ArgumentParser(description='')

        # TEST
        parser.add_argument('--test', action='store_true', help='should be True to run test.py')
        parser.add_argument('-m', '--model_path', required='--test' in sys.argv, help='the checkpoint path to test')  

        # DATASET
        parser.add_argument('--dir_hatexplain', type=str, default="./dataset", help='the root directiory of the dataset')
        
        # PRETRAINED MODEL
        model_choices = ['bert-base-uncased']
        parser.add_argument('--pretrained_model', default='bert-base-uncased', choices=model_choices, help='a pre-trained bert model to use')  

        # TRAIN
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--epochs', type=int, default=5)
        parser.add_argument('--lr', type=float, default=0.00005)
        parser.add_argument('--val_int', type=int, default=945)  
        parser.add_argument('--patience', type=int, default=3)

        ## Pre-Finetuing Task
        parser.add_argument('--intermediate', default='rp', choices=['mrp', 'rp'], required=not '--test' in sys.argv, help='choice of an intermediate task')

        ## Masked Ratioale Prediction 
        parser.add_argument('--mask_ratio', type=float, default=0.5)
        parser.add_argument('--n_tk_label', type=int, default=2)
        
        args = parser.parse_args()
        return args   

    args = get_args_1()

    dataset = HateXplainDataset(args, 'train')
    print(dataset[3])
    
    
    
    
