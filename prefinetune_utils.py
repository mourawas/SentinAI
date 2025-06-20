import torch
import random


def prepare_gts(args, max_len, bi_rats_str):
    gts = []
    for bi_rat_str in bi_rats_str:
        bi_list = bi_rat_str.split(',')
        bi_rat = [int(b) for b in bi_list]
        
        if args.intermediate == 'rp':
            bi_rat = [0]+bi_rat
            n_pads = max_len - len(bi_rat)  # num of eos + pads
            bi_gt = bi_rat + [0]*n_pads
        elif args.intermediate == 'mrp':
            bi_gt = [0]+bi_rat+[0]

        gts.append(bi_gt)

    return gts


###### MRP ######
def make_masked_rationale_label(args, labels, emb_layer, input_tokens=None, token_ambiguity=None):
    label_reps_list = []
    masked_idxs_list = []
    masked_labels_list = []
    
    #### ADDED PARTS ####
    
    # Default to random masking if token ambiguity is not provided
    use_strategic = token_ambiguity is not None and input_tokens is not None
    
    #### END OF ADDED PARTS ####
    
    for idx, label in enumerate(labels):
        idxs = list(range(len(label)))
        if args.test:
            masked_idxs = idxs[1:-1]
            masked_label = [-100]+label[1:-1]+[-100]
            label_rep = torch.zeros(len(label), emb_layer.embedding_dim)
        else:  # Validation and Training
            
            ### END OF ADDED PARTS ###
            
            if use_strategic:
                # Strategic masking based on token ambiguity
                tokens = input_tokens[idx]
                masked_idxs = []
                
                # For each token position, decide whether to mask based on token ambiguity
                for i in range(1, len(label)-1):  # Skip CLS and SEP
                    if i < len(tokens):
                        token = tokens[i]
                        # Get mask probability - default to 0.5 if token not in ambiguity dict
                        mask_prob = token_ambiguity.get(token, {}).get('mask_probability', 0.5)
                        if random.random() < mask_prob:
                            masked_idxs.append(i)
                    else:
                        # Use default masking for positions without tokens (e.g., padding)
                        if random.random() < args.mask_ratio:
                            masked_idxs.append(i)
            else:
                
                masked_idxs = random.sample(idxs[1:-1], int(len(idxs[1:-1])*args.mask_ratio))
                
            #### END OF ADDED PARTS ####
            
            masked_idxs.sort()
            label_tensor = torch.tensor(label).to(args.device)
            label_rep = emb_layer(label_tensor)
            label_rep[0] = torch.zeros(label_rep[0].shape)
            label_rep[-1] = torch.zeros(label_rep[-1].shape)
            for i in masked_idxs:
                label_rep[i] = torch.zeros(label_rep[i].shape)
            
            # For loss
            masked_label = []
            for j in idxs:
                if j in masked_idxs:
                    masked_label.append(label[j])
                else:
                    masked_label.append(-100)
            
        assert len(masked_label) == label_rep.shape[0], '[!] len(masked_label) != label_rep.shape[0] | \n{} \n{}'.format(masked_label, label_rep)
        
        masked_idxs_list.append(masked_idxs)
        masked_labels_list.append(masked_label)
        label_reps_list.append(label_rep)

    return masked_idxs_list, label_reps_list, masked_labels_list
    

def add_pads(args, max_len, labels, masked_labels, label_reps):
    assert len(labels) == len(masked_labels) == len(label_reps), '[!] add_pads | different total nums {} {} {}'.format(len(labels), len(masked_labels), len(label_reps))
    labels_pad, masked_labels_pad, label_reps_pad = [], [], []
    for label, mk_label, rep in zip(labels, masked_labels, label_reps):
        assert len(label) == len(mk_label) == rep.shape[0], '[!] add_pads | different lens of each ele {} {} {}'.format(len(label), len(mk_label), rep.shape[0])
        if args.test:
            labels_pad.append(label)
            masked_labels_pad.append(mk_label)
            label_reps_pad.append(rep)
        else:
            n_pads = max_len - len(label)
            label = label + [0]*n_pads
            mk_label = mk_label + [-100]*n_pads
            zero_ten = torch.zeros(n_pads, 768).to(args.device)
            rep = torch.cat((rep, zero_ten), 0)
            
            assert len(label) == len(mk_label) == rep.shape[0], '[!] add_pads | different lens of each ele'
            labels_pad.append(label)
            masked_labels_pad.append(mk_label)
            label_reps_pad.append(rep)

    return labels_pad, masked_labels_pad, label_reps_pad


#### ADDED PARTS ####

import os
import json
import random
import numpy as np
from utils import NumpyEncoder
import torch.nn.functional as F

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

def generate_contrastive_pairs(batch_tokens, batch_rationales, max_pairs=50):
    """ Generate positive and negative pairs for contrastive learning """
    
    positive_pairs = []
    negative_pairs = []
    
    for idx, (tokens, rationales) in enumerate(zip(batch_tokens, batch_rationales)):
        # Find indices of rationale and non-rationale tokens
        rationale_indices = [i for i, r in enumerate(rationales) if r == 1]
        non_rationale_indices = [i for i, r in enumerate(rationales) if r == 0]
        
        # Skip if no rationales or all rationales
        if not rationale_indices or not non_rationale_indices:
            continue
        
        # Generate positive pairs (rationale-rationale)
        if len(rationale_indices) > 2:
             # Limit the number of positive pairs to prevent explosion
            num_pos_pairs = min(len(rationale_indices) * (len(rationale_indices) - 1) // 2, max_pairs // 2)
            for _ in range(num_pos_pairs):
                i, j = random.sample(rationale_indices, 2)
                positive_pairs.append((idx, i, j))
                
        # Generate negative pairs (rationale-non_rationale)
        # Balance with positive pairs
        num_neg_pairs = min(len(rationale_indices) * len(non_rationale_indices), max_pairs // 2)

        for _ in range(num_neg_pairs):
            i = random.choice(rationale_indices)
            j = random.choice(non_rationale_indices)
            negative_pairs.append((idx, i, j))
            
    return positive_pairs, negative_pairs

def extract_token_representations(model_outputs, attention_mask, rationale_labels, positive_pairs, negative_pairs):
    """ Extract token-level representations and separate them into rationale/non-rationale embeddings """
    
    # Extract last few final hidden states (contextualized representations)
    all_hidden_states = torch.stack(model_outputs.hidden_states[-4:], dim=0)  # [4, batch_size, seq_len, hidden_size]
    combined_hidden_states = torch.mean(all_hidden_states, dim=0)  # [batch_size, seq_len, hidden_size]
    batch_size, seq_len, hidden_size = combined_hidden_states.shape
    
    # Create masks for valid tokens (excluding padding)
    valid_token_mask = attention_mask.bool()  # [batch_size, seq_len]
    
    # Separate rationale and non-rationale representations
    rationale_representations = []
    non_rationale_representations = []
    
    for batch_idx in range(batch_size):
        valid_tokens = valid_token_mask[batch_idx] # [seq_len]
        batch_rationales = rationale_labels[batch_idx] # [seq_len]
        batch_hidden = combined_hidden_states[batch_idx] # [seq_len, hidden_size]
        
        # Get valid token positions (exclude [CLS], [SEP], and padding)
        valid_positions = torch.where(valid_tokens)[0]
        
        for pos in valid_positions:
            token_embedding = batch_hidden[pos] # [hidden_size]
            
            if batch_rationales[pos] == 1: # Rationale token
                rationale_representations.append({
                    'embedding': token_embedding,
                    'batch_idx': batch_idx,
                    'token_idx': pos.item(),
                    'type': 'rationale'
                })
            else: # Non-rationale token
                non_rationale_representations.append({
                    'embedding': token_embedding,
                    'batch_idx': batch_idx,
                    'token_idx': pos.item(),
                    'type': 'non-rationale'
                })
                
    # Extract pair representations for contrastive learning
    positive_pair_embeddings = extract_pair_embeddings(combined_hidden_states, 
                                                      positive_pairs, pair_type='positive')
    
    negative_pair_embeddings = extract_pair_embeddings(combined_hidden_states,
                                                      negative_pairs, pair_type='negative')

    return {
        'combined_hidden_states': combined_hidden_states,
        'rationale_representations': rationale_representations,
        'non_rationale_representations': non_rationale_representations,
        'positive_pair_embeddings': positive_pair_embeddings,
        'negative_pair_embeddings': negative_pair_embeddings,
        'valid_token_mask': valid_token_mask
    }
    
def extract_pair_embeddings(hidden_states, pairs, pair_type='positive'):
    """ Extract embeddings for token pairs for contrastive learning """
    
    pair_embeddings = []
    
    for idx, token_i, token_j in pairs:
        # Extract embeddings for both tokens in the pair
        emb_i = hidden_states[idx, token_i]  # [hidden_size]
        emb_j = hidden_states[idx, token_j]  # [hidden_size]
        
        pair_embeddings.append({
        'anchor_embedding': emb_i,
        'pair_embedding': emb_j,
        'batch_idx': idx,
        'anchor_token_idx': token_i,
        'pair_token_idx': token_j,
        'pair_type': pair_type
        })
   
    return pair_embeddings

def create_contrastive_batch_tensors(positive_pairs_emb, negative_pairs_emb, device):
    """ Create batched tensors for efficient contrastive loss computation """
    
    if not positive_pairs_emb and not negative_pairs_emb:
        return None
    
    # Prepare positive pairs
    pos_anchors = []
    pos_pairs = []
    pos_labels = []
    
    for pair in positive_pairs_emb:
        pos_anchors.append(pair['anchor_embedding'])
        pos_pairs.append(pair['pair_embedding'])
        pos_labels.append(1) # Positive label
        
    # Prepare negative pairs
    neg_anchors = []
    neg_pairs = []
    neg_labels = []
    
    for pair in negative_pairs_emb:
        neg_anchors.append(pair['anchor_embedding'])
        neg_pairs.append(pair['pair_embedding'])
        neg_labels.append(0)
        
    # Combine positive and negative pairs
    all_anchors = pos_anchors + neg_anchors
    all_pairs = pos_pairs + neg_pairs
    all_labels = pos_labels + neg_labels
    
    if not all_anchors:
        return None
    
    # Convert to tensors
    anchor_tensor = torch.stack(all_anchors).to(device)  # [num_pairs, hidden_size]
    pair_tensor = torch.stack(all_pairs).to(device)      # [num_pairs, hidden_size]
    label_tensor = torch.tensor(all_labels, dtype=torch.float).to(device)  # [num_pairs]
    
    return {
        'anchors': anchor_tensor,
        'pairs': pair_tensor,
        'labels': label_tensor,
        'num_positive': len(pos_anchors),
        'num_negative': len(neg_anchors)
    }
    
def compute_contrastive_loss(anchors, pairs, labels, temperature=0.1):
    """ Compute contrastive loss for rationale prediction """
    
    if anchors.size(0) == 0:
        return torch.tensor(0.0, device=anchors.device, requires_grad=True)
  
    # Normalize embeddings
    anchors_norm = F.normalize(anchors, dim=1)
    pairs_norm = F.normalize(pairs, dim=1)
    
    # Compute cosine similarities
    similarities = torch.cosine_similarity(anchors_norm, pairs_norm, dim=1) / temperature 
    
    # Create targets for cross-entropy loss
    # Positive pairs should have high similarity, negative pairs should have low similarity
    targets = labels.float()
    
    # Use binary cross-entropy with logits
    loss = F.binary_cross_entropy_with_logits(similarities, targets)
    
    return loss
#### END OF ADDED PARTS ####