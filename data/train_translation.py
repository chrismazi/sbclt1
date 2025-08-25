import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model.sbclt_seq2seq import SBCLTEncoderDecoder
from utils import load_vocab
from config import ModelConfig
from torch.optim.lr_scheduler import get_cosine_schedule_with_warmup
import numpy as np
from evaluate_bleu import evaluate_bleu
import random
import logging
from torch.cuda.amp import autocast, GradScaler

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# -----------------------
# DATASET CLASS WITH NOISE
# -----------------------
class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, vocab, char_vocab, max_char_len=12, word_dropout=0.1):
        self.src_lines = open(src_file, encoding="utf-8").read().splitlines()
        self.tgt_lines = open(tgt_file, encoding="utf-8").read().splitlines()
        self.vocab = vocab
        self.char_vocab = char_vocab
        self.max_char_len = max_char_len
        self.word_dropout = word_dropout

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src_tokens = self.src_lines[idx].split()
        tgt_tokens = self.tgt_lines[idx].split()

        # Word dropout for source
        src_tokens = [tok if random.random() > self.word_dropout else '<unk>' for tok in src_tokens]

        src_ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in src_tokens]
        tgt_ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tgt_tokens]

        src_chars = [[self.char_vocab.get(c, 0) for c in list(tok)[:self.max_char_len]] for tok in src_tokens]
        tgt_chars = [[self.char_vocab.get(c, 0) for c in list(tok)[:self.max_char_len]] for tok in tgt_tokens]

        for chars in src_chars: chars += [0] * (self.max_char_len - len(chars))
        for chars in tgt_chars: chars += [0] * (self.max_char_len - len(chars))

        return torch.tensor(src_ids), torch.tensor(src_chars), torch.tensor(tgt_ids), torch.tensor(tgt_chars)

# -----------------------
# COLLATE FUNCTION
# -----------------------
def collate_batch(batch):
    src_ids, src_chars, tgt_ids, tgt_chars = zip(*batch)
    max_src_len = max([x.size(0) for x in src_ids])
    max_tgt_len = max([x.size(0) for x in tgt_ids])
    
    def pad(seqs, max_len, pad_val=0):
        return torch.stack([torch.cat([s, torch.zeros(max_len - s.size(0), dtype=torch.long)]) for s in seqs])

    def pad_chars(char_tensors, max_len):
        padded = []
        for ct in char_tensors:
            pad_amt = max_len - ct.size(0)
            padded.append(torch.cat([ct, torch.zeros((pad_amt, ct.size(1)), dtype=torch.long)]))
        return torch.stack(padded)

    return (
        pad(src_ids, max_src_len),
        pad_chars(src_chars, max_src_len),
        pad(tgt_ids, max_tgt_len),
        pad_chars(tgt_chars, max_tgt_len)
    )

# -----------------------
# TRAIN FUNCTION
# -----------------------
def train_translation():
    config = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler()
    grad_accum_steps = 2  # For effective batch size

    vocab = load_vocab("data/joint.vocab")
    char_vocab = {c: i+1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?!-_:;'\"")}
    char_vocab["<unk>"] = 0

    train_data = TranslationDataset("data/tokenized/train.kin.sp", "data/tokenized/train.en.sp", vocab, char_vocab, word_dropout=0.1)
    valid_data = TranslationDataset("data/tokenized/valid.kin.sp", "data/tokenized/valid.en.sp", vocab, char_vocab, word_dropout=0.0)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_data, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch)

    model = SBCLTEncoderDecoder(
        vocab_size=len(vocab),
        char_vocab_size=len(char_vocab),
        config=config
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.98)
    )

    num_training_steps = len(train_loader) * config.max_epochs // grad_accum_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=config.label_smoothing)

    best_bleu = 0
    patience_counter = 0
    global_step = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for step, (src_ids, src_chars, tgt_ids, tgt_chars) in enumerate(progress_bar):
            src_ids, src_chars = src_ids.to(device), src_chars.to(device)
            tgt_ids, tgt_chars = tgt_ids.to(device), tgt_chars.to(device)

            decoder_input = tgt_ids[:, :-1]
            decoder_target = tgt_ids[:, 1:]
            decoder_input_chars = tgt_chars[:, :-1, :]

            with autocast():
                logits = model(src_ids, src_chars, decoder_input, decoder_input_chars)
                B, T, V = logits.shape
                loss = criterion(logits.view(B*T, V), decoder_target.reshape(-1)) / grad_accum_steps

            scaler.scale(loss).backward()
            total_loss += loss.item() * grad_accum_steps

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            progress_bar.set_postfix({"loss": f"{loss.item() * grad_accum_steps:.4f}"})

        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch}: Translation Loss = {avg_loss:.4f}")

        # Validation
        model.eval()
        bleu_score = evaluate_bleu(model, valid_loader, vocab, char_vocab, config, device)
        logging.info(f"Validation BLEU: {bleu_score:.2f}")

        # Save best model
        if bleu_score > best_bleu + config.min_delta:
            best_bleu = bleu_score
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'bleu_score': bleu_score,
            }, "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logging.info(f"Early stopping triggered after {epoch} epochs")
                break

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'bleu_score': bleu_score,
        }, f"checkpoint_epoch{epoch}.pt")

if __name__ == "__main__":
    train_translation()
