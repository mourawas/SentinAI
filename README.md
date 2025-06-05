# Strategic Masking with Contrastive Multi-task Learning for Explainable Hate Speech Detection

## Enhanced Features

This repository includes several enhancements to the original MRP model [1]:

### 🎯 **Strategic Masking**
- **Strategic token masking** based on token ambiguity rather than random masking
- Improves rationale prediction by focusing on semantically important tokens
- Automatically calculates and caches token statistics for informed masking decisions

### 🔗 **Contrastive Learning**
- **Contrastive loss integration** for better token representation learning
- Learns to distinguish between positive and negative token pairs
- Configurable temperature and weighting parameters for fine-tuned control

### 🎭 **Multi-task Learning**
- **Simultaneous hate speech detection and target group identification**
- Identifies targeted groups (race, religion, gender, sexual orientation, etc.)
- Shared BERT encoder with dual classification heads for efficient learning
- 17 target group categories with multi-hot encoding support

## Initial Setting
First, Clone this git repository
```
git clone https://github.com/rayangdn/SentinAI
```
The docker commands are implemented in GNU Makefile. You can build the linux-anaconda image that the required packages are installed. The details are written in Dockerfile and Makefile. Or you can use the docker commands directly. 
1. Set the name variables in Makefile, such as IMAGE_NAME, IMAGE_TAG, CONTAINER_NAME, CONTAINER_PORT, NVIDIA_VISIBLE_DEVICES.
2. Build the docker image. Use the Makefile command on the directory where the Dockerfile located.
```
make docker-build
```
3. Run the docker container from the built image.
```
make docker-run
```
4. Execute the container.
```
make docker-exec
```

## Models
If you would like to run the models, you can finde them and download them in this Google drive [link](https://drive.google.com/drive/u/1/folders/1vAkw4C90RV8_fSjDZRCvOLHtRD4-zHtj). This contains two folders:
- **finetune_1st**: includes the checkpoints of the pre-finetuned models. Each of names shows what's the pre-finetuning method and some infos of the hyperparameters.
```
📁finetune_1st
 ╰─ 📁{checkpoint name}
     ╰─ checkpoint
```
- **finetune_2nd**: if you would like to run the final hate speech detection models, you only need the finetune_2nd and don't have to get finetune_1st. it includes the checkpoints of the final models finetuned on hate speech detection. The upper folders indicate which pre-finetuned parameter was used for intialization among the checkpoints in the finetune_1st folder. Each of pre-finetuned checkpoints was finetuned on both two and three-class classification for hate speech detection according to HateXplain benchmark. The two classes are *non-toxic* and *toxic*, and the three classes are *normal*, *offensive*, and *hate speech*.
```
📁finetune_2nd
 ╰─ 📁{the pre-finetuned checkpoint name}
     ╰─ 📁{the checkpoint name}
         ╰─ checkpoint 
```

## Test
For testing a model, run second_test.py like below:
```python
python second_test.py -m {model path to test}
```

If you run a model which trained on two-class detection, it would be tested for Bias-based metrics of hateXplain benchmark. And a model which trained on three-class detection, you could get the results for Performance-based metrics and Explainability-based metrics.

**Note:** Multi-task models are automatically detected based on the model path containing *multitask* and will additionally report target group identification metrics.

## Train

### Pre-finetuning (First Stage)
Train intermediate models with enhanced features:
```python
python first_train.py --intermediate {mrp|rp} [--strategic_masking] [--contrastive_loss] [additional options]
```
**Enhanced Arguments:**
- `--strategic_masking`: Enable intelligent token masking based on ambiguity
- `--contrastive_loss`: Enable contrastive learning for better representations
- `--contrastive_weight`: Weight for contrastive loss (default: 0.1)
- `--contrastive_temperature`: Temperature for contrastive loss (default: 0.1)

### Final Training (Second Stage)
Train hate speech detection models with optional multi-task learning:
```python
python second_train.py -pf_m {pre-finetuned model path} --num_labels {2|3} [--multitask] [additional options]
```
**Enhanced Arguments:**
- `--multitask`: Enable multi-task learning for target group identification
- `--alpha`: Weight for hate speech detection loss in multi-task learning (default: 0.7)

**Examples:**
```bash
# Standard training
python second_train.py -pf_m bert-base-uncased --num_labels 3

# Multi-task training with custom loss weighting
python second_train.py -pf_m ./finetune_1st/checkpoint --num_labels 3 --multitask --alpha 0.8

# Pre-finetuning with strategic masking and contrastive learning
python first_train.py --intermediate mrp --strategic_masking --contrastive_loss --contrastive_weight 0.15
```

## Enhanced Model Outputs

### Multi-task Models
When using `--multitask`, models will output:
- **Hate speech classification** (normal/offensive/hatespeech)
- **Target group identification** (17 categories including race, religion, gender, etc.)
- Combined evaluation metrics for both tasks

### Strategic Masking
- Automatically generates `token_statistics.json` for optimized masking
- Improves rationale prediction accuracy by focusing on ambiguous tokens

### Contrastive Learning
- Enhanced token representations through positive/negative pair learning
- Configurable loss weighting and temperature parameters

## References
**[1] Masked Rationale Prediction for Explainable Hate Speech Detection:** [COLING](https://aclanthology.org/2022.coling-1.577/)  |  [arXiv](https://arxiv.org/abs/2211.00243) | [GitHub](https://github.com/alatteaday/mrp_hate-speech-detection)   
**[2] HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection**: [arXiv](https://arxiv.org/abs/2012.10289) | [GitHub](https://github.com/hate-alert/HateXplain)
 
