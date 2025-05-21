import torch
from torch.utils.data import WeightedRandomSampler
import os
import numpy as np
import random
import json


def get_device():
    if torch.cuda.is_available():
        print("device = cuda")
        return torch.device('cuda')
    else:
        print("device = cpu")
        return torch.device('cpu')


def add_tokens_to_tokenizer(args, tokenizer):
    special_tokens_dict = {'additional_special_tokens': 
                            ['<user>', '<number>']}  # hatexplain
    n_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    # print(tokenizer.all_special_tokens) 
    # print(tokenizer.all_special_ids)
    
    return tokenizer


class GetLossAverage(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()  # type -> int
        v = v.data.sum().item()  # type -> float
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def aver(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def get_weighted_sampler(args, dataset):
    cls_count = dataset.label_count
    clses = list(range(len(dataset.label_list)))
    total_count = len(dataset)

    class_weights = [total_count / cls_count[i] for i in range(len(cls_count))] 
    weights = [class_weights[clses[i]] for i in range(int(total_count))] 
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(total_count))

    return sampler 
    

def save_checkpoint(args, losses, model_state, trained_model):
    # checkpoint = {
    #     'args': args,
    #     'model_state': model_state,
    #     'optimizer_state': optimizer_state
    # }
    file_name = args.exp_name + '.ckpt'
    trained_model.save_pretrained(save_directory=os.path.join(args.dir_result, file_name))

    args.waiting += 1
    if losses[-1] <= min(losses):
        # print(losses)
        args.waiting = 0
        file_name = 'BEST_' + file_name
        trained_model.save_pretrained(save_directory=os.path.join(args.dir_result, file_name))
        
        if args.intermediate == 'mrp':
            # Save the embedding layer params
            emb_file_name = args.exp_name + '_emb.ckpt'
            torch.save(model_state.state_dict(), os.path.join(args.dir_result, emb_file_name))

        print("[!] The best checkpoint is updated")


def get_token_rationale(tokenizer, text, rationale, id):
    text_token = tokenizer.tokenize(' '.join(text))
    assert len(text) == len(rationale), '[!] len(text) != len(rationale) | {} != {}\n{}\n{}'.format(len(text), len(rationale), text, rationale)
    
    rat_token = []
    for t, r in zip(text, rationale):
        token = tokenizer.tokenize(t)
        rat_token += [r]*len(token)
    assert len(text_token) == len(rat_token), "#token != #target rationales of {}".format(id)
    return rat_token


def make_final_rationale(id, rats_list):
    rats_np = np.array(rats_list)
    sum_np = rats_np.sum(axis=0)
    try:
        avg_np = sum_np / len(rats_list)
        avg_rat = avg_np.tolist()
        bi_rat = []
        for el in avg_rat:
            if el >= 0.5:
                bi_rat.append(1)
            else:
                bi_rat.append(0)
    except:
        print(id)
    
    return avg_rat, bi_rat


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
    
#### ADDED PARTS ####
    
def calculate_token_statistics(dataset, tokenizer, save_path=None):
    """ Calculate token statistics for strategic masking """
    
    # Track token distribution across classes
    token_class_dist = {}  # Maps tokens to count in each class
    token_rationale_count = {}  # How often token appears as rationale
    token_total_count = {}  # Total token occurrences
    class_counts = {'hatespeech': 0, 'normal': 0, 'offensive': 0}
    
    print("Analyzing token distribution across classes...")
    for i in range(len(dataset)):
        text, cls_num, fin_rat_str = dataset[i]
        tokens = tokenizer.tokenize(text)
        label = ['hatespeech', 'normal', 'offensive'][cls_num]
        class_counts[label] += 1
        
        # Parse rationales
        rationales = []
        if fin_rat_str:
            rationales = [int(r) for r in fin_rat_str.split(',')]
        
        for token_idx, token in enumerate(tokens):
            # Skip special tokens
            if token in ['[CLS]', '[SEP]', '[PAD]', '<user>', '<number>']:
                continue
            
            # Initialize token stats if needed
            if token not in token_class_dist:
                token_class_dist[token] = {'hatespeech': 0, 'normal': 0, 'offensive': 0}
                token_rationale_count[token] = 0
                token_total_count[token] = 0
            
            # Update counts
            token_class_dist[token][label] += 1
            token_total_count[token] += 1
            
            # Count rationale occurrences
            if token_idx < len(rationales) and rationales[token_idx] == 1:
                token_rationale_count[token] += 1    
            
    print("Calculating masking probabilities...")
    # Calculate token-level metrics
    token_metrics = {}
    total_docs = sum(class_counts.values())
    
    for token, counts in token_class_dist.items():
        token_count = token_total_count[token]
        if token_count < 5: # Skip rare tokens
            continue
        
        # Calculate class distribution probabilities
        probs = {}
        for cls, count in counts.items():
            # Calculate P(class|token) - probability of class given token 
            if token_count >= 0:
                probs[cls] = count / token_count
            else:
                probs[cls] = 0.0
                
            # Calculate entropy (higer = more ambiguous)
            non_zero_probs = [p for p in probs.values() if p > 0]
            entropy = -sum(p * np.log(p) for p in non_zero_probs) if non_zero_probs else 0
            
            # Normalize entropy by max possible entropy (log of number of classes)
            max_entropy = np.log(len(class_counts))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            # Calculate PMI (Pointwise Mutual Information) for each class
            pmi = {}
            for cls, count in counts.items():
                # P(class) - prior probability of class
                p_class = class_counts[cls] / total_docs
                
                # P(token, class) - joint probability
                p_token_class = count / total_docs
                
                # P(token) - probability of token
                p_token = token_count / total_docs
                
                # PMI = log(P(token, class) / (P(token) * P(class)))
                # High absolute PMI means strong association (positive or negative)
                if p_token > 0 and p_class > 0 and p_token_class > 0:
                    pmi[cls] = np.log(p_token_class / (p_token * p_class))
                else:
                    pmi[cls] = 0.0
                    
            # Calculate absolute PMI values (higher = stronger association with specific class)
            abs_pmi_values = [abs(val) for val in pmi.values()]
            avg_abs_pmi = sum(abs_pmi_values) / len(abs_pmi_values) if abs_pmi_values else 0
            
            # Calculate rationale ratio (how often token is part of rationale)
            rationale_ratio = token_rationale_count[token] / token_count if token_count > 0 else 0
            
            # Term frequency - inverse document frequency (TF-IDF) like measure
            # Lower for common words across all documents
            idf = np.log(total_docs / (token_count + 1))
            
            # Determine masking probability based on our metrics:
            # 1. Higher entropy (ambiguity) -> higher masking probability
            # 2. Higher absolute PMI (strong class association) -> lower masking probability for common words
            # 3. Higher rationale ratio -> lower masking probability (we want to keep obvious hate indicators)
            # 4. Higher IDF (rare words) -> lower masking probability
            
            # Base probability (0.3-0.7) adjusted by metrics
            base_prob = 0.5
            entropy_factor = normalized_entropy * 0.4 # Boost for ambiguous tokens
            pmi_factor = -min(avg_abs_pmi, 0.15) * 0.3 # Reduction for class-associated tokens
            rationale_factor = -min(rationale_ratio * 0.3, 0.3)  # Reduction for rationale tokens
            common_word_factor = -min(1.0 / (idf + 1), 0.3)  # Reduction for very common words
            
            # Special handling for stopwords and very common words
            is_common_word = token_count > 1000 and normalized_entropy < 0.1
            
            if is_common_word:
                # For very common words with low entropy, use higher probability
                mask_probability = min(0.9, base_prob + 0.3)
            else:
                # For normal tokens, combine all factors
                mask_probability = min(0.9, max(0.1, base_prob + entropy_factor + pmi_factor + rationale_factor + common_word_factor))
        
            token_metrics[token] = {
                'entropy': entropy,
                'normalized_entropy': normalized_entropy,
                'pmi': pmi,
                'avg_abs_pmi': avg_abs_pmi,
                'rationale_ratio': rationale_ratio,
                'count': token_count,
                'idf': idf,
                'class_distribution': {cls: count/token_count for cls, count in counts.items()},
                'mask_probability': mask_probability
            }
                
    # Print some statistics for inspection
    print("\nTop tokens by masking probability:")
    tokens_by_prob = sorted([(t, m['mask_probability']) for t, m in token_metrics.items()], 
                        key=lambda x: x[1], reverse=True)
    for token, prob in tokens_by_prob[:3]:
        metrics = token_metrics[token]
        print(f"{token}: prob={prob:.3f}, entropy={metrics['normalized_entropy']:.3f}, "
            f"rationale_ratio={metrics['rationale_ratio']:.3f}, count={metrics['count']}")
    
    print("\nTop ambiguous tokens:")
    ambiguous_tokens = sorted([(t, m['normalized_entropy']) for t, m in token_metrics.items()], 
                            key=lambda x: x[1], reverse=True)
    for token, entropy in ambiguous_tokens[:3]:
        metrics = token_metrics[token]
        print(f"{token}: entropy={entropy:.3f}, prob={metrics['mask_probability']:.3f}, "
            f"count={metrics['count']}, distribution={metrics['class_distribution']}")
    
    print("\nTop rationale tokens:")
    rationale_tokens = sorted([(t, m['rationale_ratio']) for t, m in token_metrics.items() 
                            if m['count'] > 20], key=lambda x: x[1], reverse=True)
    for token, ratio in rationale_tokens[:3]:
        metrics = token_metrics[token]
        print(f"{token}: rationale_ratio={ratio:.3f}, prob={metrics['mask_probability']:.3f}, "
            f"count={metrics['count']}")
        
    # Save statistics to file for reuse
    output_file = os.path.join(save_path, 'token_statistics.json')
    with open(output_file, 'w') as f:
        json.dump(token_metrics, f, cls=NumpyEncoder)
    
    print(f"\nToken statistics saved to {output_file}")
    return token_metrics

def set_seed(seed=42):
    """ Set random seed for reproducibility """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
#### END OF ADDED PARTS ####
