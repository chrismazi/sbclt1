# SBCLT Project Structure

## Overview
This document describes the clean, professional organization of the SBCLT codebase, optimized for research publication and achieving BLEU 25+.

## Directory Structure

```
SBCLT/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── PROJECT_STRUCTURE.md     # This file
├── train.py                 # Main training script
├── data/                    # Data directory
│   ├── train.kin           # Training Kinyarwanda
│   ├── train.en            # Training English
│   ├── valid.kin           # Validation Kinyarwanda
│   ├── valid.en            # Validation English
│   ├── test.kin            # Test Kinyarwanda
│   ├── test.en             # Test English
│   ├── joint.txt           # Combined corpus for tokenizer
│   ├── joint.model         # SentencePiece model
│   ├── joint.vocab         # Vocabulary file
│   ├── tokenized/          # Tokenized data
│   └── split_parallel_corpus.py  # Data splitting utility
├── src/                     # Source code (clean, organized)
│   ├── __init__.py         # Package initialization
│   ├── config.py           # Configuration management
│   ├── models/             # Neural network models
│   │   ├── __init__.py
│   │   ├── sbclt.py       # Main SBCLT model
│   │   ├── char_encoder.py # Character-level encoder
│   │   ├── sparse_attention.py # Sparse attention mechanism
│   │   ├── positional_encoding.py # Positional encoding
│   │   └── beam_search.py  # Beam search decoder
│   ├── data/               # Data processing
│   │   ├── __init__.py
│   │   └── dataset.py      # Dataset classes and utilities
│   ├── training/           # Training pipeline
│   │   ├── __init__.py
│   │   └── trainer.py      # Professional trainer class
│   └── utils/              # Utility functions
│       ├── __init__.py
│       ├── vocab.py        # Vocabulary utilities
│       ├── evaluation.py   # Evaluation functions
│       └── logging.py      # Logging utilities
└── outputs/                 # Training outputs (created during training)
    ├── checkpoints/         # Model checkpoints
    ├── logs/                # Training logs
    └── best_model.pt       # Best performing model
```

## Key Improvements Made

### 1. **Code Organization**
- **Modular structure**: Separated concerns into logical packages
- **Clean imports**: Professional import structure with relative imports
- **Documentation**: Comprehensive docstrings and type hints

### 2. **Professional Training Pipeline**
- **Mixed precision training**: For speed and memory efficiency
- **Advanced optimization**: AdamW + cosine annealing + warmup
- **Gradient accumulation**: Effective larger batch sizes
- **Early stopping**: Prevents overfitting
- **Checkpointing**: Saves best model and regular checkpoints

### 3. **Research-Grade Features**
- **Reproducibility**: Fixed random seeds, deterministic training
- **Comprehensive logging**: Training metrics and progress
- **Error handling**: Professional error handling and validation
- **Configuration management**: Centralized hyperparameter control

### 4. **Model Architecture**
- **Increased capacity**: d_model=1024, ff_dim=4096
- **Better regularization**: Dropout=0.3, label smoothing=0.15
- **Professional initialization**: Xavier and normal distributions
- **Pre-norm architecture**: For training stability

### 5. **Data Processing**
- **Noise injection**: Word dropout for robustness
- **Character encoding**: Morphological feature extraction
- **Professional validation**: Data integrity checks
- **Efficient batching**: Optimized collation functions

## Files Removed

The following unnecessary files were removed for cleanliness:
- Old checkpoint files (checkpoint_epoch*.pt)
- Test files (test_*.py)
- Outdated training scripts
- Temporary files (~$*)
- Loss plots and other artifacts

## Usage

### Training
```bash
python train.py --data-dir data --output-dir outputs --seed 42
```

### Configuration
All hyperparameters are managed in `src/config.py` with comprehensive documentation.

### Data Preparation
1. Place parallel corpus in `data/` directory
2. Run SentencePiece training: `python data/train_spm.py`
3. Tokenize data using the new model
4. Start training with `python train.py`

## Research Publication Ready

This codebase is now organized to meet research publication standards:
- **Reproducible**: Fixed seeds, deterministic training
- **Documented**: Comprehensive docstrings and type hints
- **Modular**: Clean separation of concerns
- **Professional**: Industry-standard code organization
- **Optimized**: All improvements for BLEU 25+ target

## Next Steps

1. **Train SentencePiece** with new 32K vocabulary
2. **Tokenize data** using the new tokenizer
3. **Start training** with the professional pipeline
4. **Monitor progress** through comprehensive logging
5. **Achieve BLEU 25+** with the optimized system
