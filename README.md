# SBCLT: Sparse Bayesian Cross-Lingual Transformer

A professional neural machine translation (NMT) system for Kinyarwanda-English, featuring:
- Transformer with sparse Bayesian attention
- Character-level and subword-level encoding
- Advanced training and evaluation pipeline

## Features
- Mixed precision training for speed and stability
- Beam search decoding
- Detokenized BLEU evaluation
- Back-translation for data augmentation
- Reproducible experiments with logging and random seed control

---

## Setup

1. **Install dependencies**
   ```bash
   pip install torch sentencepiece tqdm numpy nltk
   ```

2. **Prepare Data**
   - Place your parallel corpus in `data/train.kin`, `data/train.en`, `data/valid.kin`, `data/valid.en`.
   - (Optional) Add monolingual English data as `data/monolingual.en` for back-translation.

3. **Train SentencePiece Tokenizer**
   ```bash
   python data/train_spm.py
   ```
   This will create `data/joint.model` and `data/joint.vocab` with a 32,000 BPE vocabulary.

4. **Tokenize Data**
   Use your own script or adapt `data/tokenize_with_spm.py` to tokenize all train/valid files with the new model.

---

## Training

1. **Train the Model**
   ```bash
   python data/train_translation.py
   ```
   - Uses mixed precision, logging, and early stopping.
   - Best model is saved as `best_model.pt`.

2. **Monitor Training**
   - Logs are printed to the console.
   - Checkpoint files are saved for each epoch.

---

## Evaluation

1. **Evaluate BLEU Score**
   ```bash
   python data/evaluate_bleu.py
   ```
   - Uses beam search and detokenized BLEU.
   - Prints sample translations and logs BLEU score.

---

## Back-Translation (Data Augmentation)

1. **Generate Synthetic Parallel Data**
   - Place monolingual English sentences in `data/monolingual.en`.
   - Run:
     ```bash
     python data/back_translate.py
     ```
   - Outputs synthetic Kinyarwanda (`data/backtrans.kin`) and the original English (`data/backtrans.en`).

2. **Retrain with Augmented Data**
   - Concatenate `backtrans.kin`/`backtrans.en` to your original training files.
   - Retokenize and retrain for further BLEU improvement.

---

## Best Practices
- Set random seeds for reproducibility (already handled in scripts).
- Use a GPU for training and inference.
- Monitor BLEU and loss for overfitting.
- Use early stopping and save the best model.
- Experiment with hyperparameters in `data/config.py` for best results.

---

## Citation
If you use this codebase, please cite the original author and this repository.

---

## Contact
For questions or contributions, please open an issue or pull request. 