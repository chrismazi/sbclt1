"""
Professional training script for SBCLT: Sparse Bayesian Cross-Lingual Transformer

This module implements a research-grade training pipeline with:
- Mixed precision training
- Advanced optimization strategies
- Comprehensive logging and monitoring
- Early stopping and checkpointing
- Reproducible experiments
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import random
import numpy as np
from pathlib import Path

from ..models.sbclt import SBCLTEncoderDecoder
from ..data.dataset import TranslationDataset, collate_batch
from ..config import ModelConfig
from ..utils.vocab import load_vocab, create_char_vocab
from ..utils.evaluation import evaluate_bleu
from ..utils.logging import setup_logging, log_metrics


class SBCLTTrainer:
    """
    Professional trainer for SBCLT model with research-grade features.
    
    Features:
    - Mixed precision training with gradient scaling
    - Advanced optimization (AdamW + cosine annealing)
    - Comprehensive logging and monitoring
    - Early stopping with patience
    - Checkpoint management
    - Reproducible experiments
    """
    
    def __init__(self, config: ModelConfig, data_dir: str, output_dir: str):
        """
        Initialize the trainer.
        
        Args:
            config: Model configuration
            data_dir: Directory containing training data
            output_dir: Directory for saving outputs
        """
        self.config = config
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup device and reproducibility
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_reproducibility()
        
        # Setup logging
        setup_logging(self.output_dir / "training.log")
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._setup_data()
        self._setup_model()
        self._setup_optimization()
        
        # Training state
        self.best_bleu = 0
        self.patience_counter = 0
        self.global_step = 0
        
        self.logger.info(f"Trainer initialized on {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _setup_reproducibility(self):
        """Setup for reproducible experiments."""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _setup_data(self):
        """Setup data loaders and vocabularies."""
        # Load vocabularies
        self.vocab = load_vocab(self.data_dir / "joint.vocab")
        self.char_vocab = create_char_vocab()
        
        # Create datasets
        train_data = TranslationDataset(
            self.data_dir / "tokenized" / "train.kin.sp",
            self.data_dir / "tokenized" / "train.en.sp",
            self.vocab, self.char_vocab, word_dropout=0.1
        )
        valid_data = TranslationDataset(
            self.data_dir / "tokenized" / "valid.kin.sp",
            self.data_dir / "tokenized" / "valid.en.sp",
            self.vocab, self.char_vocab, word_dropout=0.0
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_data, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            collate_fn=collate_batch,
            num_workers=4
        )
        self.valid_loader = DataLoader(
            valid_data, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            collate_fn=collate_batch,
            num_workers=4
        )
        
        self.logger.info(f"Training samples: {len(train_data):,}")
        self.logger.info(f"Validation samples: {len(valid_data):,}")
    
    def _setup_model(self):
        """Setup the SBCLT model."""
        self.model = SBCLTEncoderDecoder(
            vocab_size=len(self.vocab),
            char_vocab_size=len(self.char_vocab),
            config=self.config
        ).to(self.device)
        
        # Mixed precision setup
        if self.config.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def _setup_optimization(self):
        """Setup optimizer, scheduler, and loss function."""
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.98)
        )
        
        # Scheduler
        num_training_steps = len(self.train_loader) * self.config.max_epochs // self.config.gradient_accumulation_steps
        self.scheduler = torch.optim.lr_scheduler.get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=0, 
            label_smoothing=self.config.label_smoothing
        )
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {epoch}", 
            leave=False
        )
        
        for step, (src_ids, src_chars, tgt_ids, tgt_chars) in enumerate(progress_bar):
            # Move to device
            src_ids, src_chars = src_ids.to(self.device), src_chars.to(self.device)
            tgt_ids, tgt_chars = tgt_ids.to(self.device), tgt_chars.to(self.device)
            
            # Prepare decoder input/output
            decoder_input = tgt_ids[:, :-1]
            decoder_target = tgt_ids[:, 1:]
            decoder_input_chars = tgt_chars[:, :-1, :]
            
            # Forward pass with mixed precision
            if self.config.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    logits = self.model(src_ids, src_chars, decoder_input, decoder_input_chars)
                    B, T, V = logits.shape
                    loss = self.criterion(logits.view(B*T, V), decoder_target.reshape(-1))
                    loss = loss / self.config.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
                total_loss += loss.item() * self.config.gradient_accumulation_steps
            else:
                logits = self.model(src_ids, src_chars, decoder_input, decoder_input_chars)
                B, T, V = logits.shape
                loss = self.criterion(logits.view(B*T, V), decoder_target.reshape(-1))
                loss = loss / self.config.gradient_accumulation_steps
                
                loss.backward()
                total_loss += loss.item() * self.config.gradient_accumulation_steps
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0 or (step + 1) == num_batches:
                if self.config.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip
                )
                
                if self.config.use_mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item() * self.config.gradient_accumulation_steps:.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Logging
            if step % self.config.log_interval == 0:
                log_metrics({
                    "epoch": epoch,
                    "step": step,
                    "loss": loss.item() * self.config.gradient_accumulation_steps,
                    "learning_rate": self.scheduler.get_last_lr()[0]
                })
        
        return total_loss / num_batches
    
    def validate(self, epoch: int):
        """Validate the model."""
        self.model.eval()
        
        with torch.no_grad():
            bleu_score = evaluate_bleu(
                self.model, 
                self.valid_loader, 
                self.vocab, 
                self.char_vocab, 
                self.config, 
                self.device
            )
        
        self.logger.info(f"Epoch {epoch} - Validation BLEU: {bleu_score:.2f}")
        
        return bleu_score
    
    def save_checkpoint(self, epoch: int, bleu_score: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'bleu_score': bleu_score,
            'config': self.config,
            'global_step': self.global_step
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with BLEU: {bleu_score:.2f}")
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        for epoch in range(1, self.config.max_epochs + 1):
            # Training
            train_loss = self.train_epoch(epoch)
            self.logger.info(f"Epoch {epoch} - Training Loss: {train_loss:.4f}")
            
            # Validation
            if epoch % self.config.eval_interval == 0:
                bleu_score = self.validate(epoch)
                
                # Check if this is the best model
                is_best = bleu_score > self.best_bleu + self.config.min_delta
                if is_best:
                    self.best_bleu = bleu_score
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Save checkpoint
                if self.config.save_checkpoints:
                    self.save_checkpoint(epoch, bleu_score, is_best)
                
                # Early stopping
                if self.patience_counter >= self.config.patience:
                    self.logger.info(f"Early stopping triggered after {epoch} epochs")
                    break
        
        self.logger.info(f"Training completed. Best BLEU: {self.best_bleu:.2f}")


def main():
    """Main training function."""
    config = ModelConfig()
    
    # Setup paths
    data_dir = "data"
    output_dir = "outputs"
    
    # Create trainer and start training
    trainer = SBCLTTrainer(config, data_dir, output_dir)
    trainer.train()


if __name__ == "__main__":
    main()
