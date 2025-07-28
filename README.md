# Fine-tuning BERT for Greek Multiword Expression Identification

This repository contains the implementation and resources for fine-tuning BERT models for Masked Language Modeling and identification of sentences with Verbal Multiword Expressions (MWEs) in Modern Greek.

## Overview

This research investigates the effectiveness of BERT models for classifying sentences based on whether they contain Greek multiword expressions. The study adopts a two-phase methodology:

1. **Phase 1**: Fine-tuning a Greek BERT model for masked language modeling using 4,406 MWEs from lexical resources
2. **Phase 2**: Further fine-tuning and evaluating the model's performance on Greek sentences from web sources

## Key Contributions

- **Improved MWE Classification**: Fine-tuned Greek BERT achieved 80% accuracy vs 70.5% for baseline
- **Domain Knowledge Acquisition**: Model perplexity improved from 303.21 to 3.81 after fine-tuning
- **Structured Lexical Resource Integration**: Demonstrated effectiveness of using structured lexical resources for free text classification
- **Cross-linguistic Applicability**: Methodology applicable to other languages with similar lexical resources

## Results

| Model | Accuracy | Precision (Literal) | Precision (MWE) | F1-Score (Literal) | F1-Score (MWE) |
|-------|----------|-------------------|-----------------|-------------------|----------------|
| **MWE-fine-tuned Greek BERT** | **79.5%** | **83%** | **77%** | **78%** | **81%** |
| Original Greek BERT | 70.5% | 72% | 69% | 69% | 71% |
| Logistic Regression | 69.5% | 68% | 71% | 71% | 68% |

## Dataset Structure

### PolylexMG Dataset
The project uses the **PolylexMG** dataset, containing 5,635 Modern Greek verbal multiword expressions organized into 20 syntactic tables based on Lexicon-Grammar theory.

#### Dataset Components:

1. **Full-Expression Subdataset** (4,406 entries)
   - Contains MWEs in canonical form (present tense, first person)
   - Used for masked language modeling training
   - Format: Subject-Verb-Object structure

2. **Classification-Task Subdataset** (200 entries)
   - Contains real-world sentences with/without MWEs
   - Used for sentence classification training
   - Labels: 1 (contains MWE), 0 (literal usage)

#### Example Entries:

**Full-Expression Subdataset:**
```
ID | Expression | Meaning
1  | αδειάζω τη γωνιά σε | 'Make room for'
3  | αλλάζω τον αδόξαστο σε | 'Inflict severe blow to'
```

**Classification-Task Subdataset:**
```
Sentence | Meaning | Label
Πλήρωσε μόνο για το δείπνο του | 'He paid only for his meal' | 0
Σ' αυτή τη φάση, κάθε πόντος... | 'At this phase, every point...' | 1
```

## Architecture

### Phase 1: Masked Language Modeling
- **Base Model**: Greek BERT (`nlpaueb/bert-base-greek-uncased-v1`)
- **Training Data**: 17,624 examples (4,406 unique × 4 augmented versions)
- **Masking Strategy**: 25% probability per token
- **Output**: MWE-aware language model

### Phase 2: Sentence Classification
- **Base Model**: MWE-fine-tuned Greek BERT from Phase 1
- **Architecture**: BERT + classification head
- **Training**: 10-fold cross-validation
- **Output**: Binary classification (MWE vs. literal)

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Polylex-Text-classification-BERT
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download the fine-tuned model:**
```bash
# The model is available on Hugging Face Hub
# https://huggingface.co/polylexmg/bert-base-greek-uncased-v6-finetuned-polylex-mg
```

## Usage

### 1. Masked Language Modeling

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# Load the fine-tuned model
model_name = "polylexmg/bert-base-greek-uncased-v6-finetuned-polylex-mg"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Example usage
text = "ανοίγω το [MASK]"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
predictions = torch.topk(outputs.logits[0, 1, :], 5)
```

### 2. Sentence Classification

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the classification model
model_name = "your-fine-tuned-classification-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Classify a sentence
text = "Σ' αυτή τη φάση, κάθε πόντος είναι σημαντικό"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=100)
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits).item()
# 1 = contains MWE, 0 = literal usage
```

### 3. Training Your Own Model

#### Phase 1: Masked Language Modeling
```python
# See Fine-tuning a masked language model (PyTorch) in Polylex.ipynb
# Key parameters:
# - Learning rate: 0.00005
# - Batch size: 512
# - Epochs: 500
# - Weight decay: 0.01
```

#### Phase 2: Classification
```python
# See Text_classification_with_BERT_in_PyTorch_for_Polylex.ipynb
# Key parameters:
# - Learning rate: 1e-6
# - Batch size: 64
# - Max sequence length: 100
# - Cross-validation: 10-fold
```

## Evaluation

### Qualitative Evaluation
The model was tested on 14 verbal constructs with masked nouns. Results show:
- Fine-tuned model correctly identifies idiomatic expressions
- Original Greek BERT produces incomplete or non-idiomatic completions

### Quantitative Evaluation
- **Perplexity Improvement**: 303.21 → 3.81 (after fine-tuning)
- **Classification Accuracy**: 79.5% (vs 70.5% baseline)
- **Cross-validation**: 10-fold with consistent performance

## Technical Details

### Model Specifications
- **Base Architecture**: BERT-BASE-UNCASED (12 layers, 768 hidden units, 12 attention heads)
- **Parameters**: ~110 million
- **Vocabulary**: 35k subword BPE tokens
- **Training**: Adam optimizer, linear learning rate schedule with warmup

### Data Augmentation Strategy
- **Original**: 4,406 unique MWEs
- **Augmented**: 17,624 training examples
- **Method**: 25% masking probability per token
- **Variations**: Multiple masking positions per expression

### Training Configuration
```python
# Phase 1 (MLM)
GRADIENT_ACCUMULATION_STEPS = 2
NUM_TRAIN_EPOCHS = 500
BATCH_SIZE = 512
LEARNING_RATE = 0.00005
WEIGHT_DECAY = 0.01

# Phase 2 (Classification)
NUM_TRAIN_EPOCHS = 80
BATCH_SIZE = 64
LEARNING_RATE = 0.000001
MAX_SEQ_LENGTH = 100
```

## Performance Analysis

### Strengths
- **Domain Knowledge**: Model learns MWE-specific patterns
- **Generalization**: Performs well on unseen MWE contexts
- **Robustness**: Consistent performance across cross-validation folds

### Limitations
- **Dataset Size**: Limited to 4,406 unique MWEs
- **Context Dependency**: Some MWEs may have both literal and idiomatic readings
- **Language Specificity**: Results specific to Modern Greek

## Research Context

### Lexicon-Grammar Framework
The methodology is based on Lexicon-Grammar theory, which:
- Classifies MWEs by syntactic structure
- Encodes distributional and semantic properties
- Enables cross-linguistic comparisons

### MWE Types Covered
- **Fixed Expressions**: Complete idiomatic phrases
- **Semi-fixed Phrases**: Allow some variation
- **Support Verb Constructions**: Light verb + noun combinations

## Citation

If you use this work, please cite:

```bibtex
@article{fotopoulou2024fine,
  title={Fine-tuning BERT for Masked Language Modelling and Identification of Sentences with Verbal Multiword Expressions in the Modern Greek language},
  author={Fotopoulou, Aggeliki and Kyriazi, Panagiota and Nousias, Stavros},
  journal={Computational Linguistics},
  year={2024}
}
```

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **PolylexMG Dataset**: Comprehensive Greek MWE resource
- **Greek BERT**: Base model by NLPAUEB
- **Hugging Face**: Transformers library and model hosting
- **PARSEME Network**: MWE research community

## Contact

For questions or collaboration:
- **Aggeliki Fotopoulou**: Institute for Language and Speech Processing, Athena Research Center
- **Panagiota Kyriazi**: Institute for Language and Speech Processing, Athena Research Center  
- **Stavros Nousias**: School of Engineering and Design, Technical University of Munich

## Links

- **Paper**: [Link to paper]
- **Model**: [Hugging Face Hub](https://huggingface.co/polylexmg/bert-base-greek-uncased-v6-finetuned-polylex-mg)
- **Dataset**: [PolylexMG Resource](https://polylexmg.ilsp.gr)
- **Code**: [GitHub Repository]
- **Google Colab Notebooks**: [Google Drive](https://drive.google.com/drive/folders/1bpSE2qGOnqIRjhVzx3doEOtUbqm-hSAe)

---

*This research contributes to the understanding of how transformer-based models can be optimized to identify and process complex linguistic structures, particularly in the context of Greek language processing.* 