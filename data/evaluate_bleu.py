import torch
from torch.utils.data import DataLoader
from model.sbclt_seq2seq import SBCLTEncoderDecoder
from utils import load_vocab
import sentencepiece as spm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from train_translation import TranslationDataset, collate_batch
from config import ModelConfig
from model.beam_search import beam_search
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def evaluate_bleu(model, dataloader, vocab, char_vocab, config, device, sp):
    model.eval()
    refs = []
    hyps = []
    with torch.no_grad():
        for src_ids, src_chars, tgt_ids, tgt_chars in dataloader:
            src_ids, src_chars = src_ids.to(device), src_chars.to(device)
            # Use beam search for decoding
            pred_ids = []
            for i in range(src_ids.size(0)):
                pred = beam_search(
                    model=model,
                    src_ids=src_ids[i:i+1],
                    src_chars=src_chars[i:i+1],
                    vocab=vocab,
                    char_vocab=char_vocab,
                    config=config,
                    device=device
                )
                pred_ids.append(pred)
            # Convert predictions and references to detokenized strings
            for pred, tgt in zip(pred_ids, tgt_ids):
                pred_tokens = [k for k, v in vocab.items() if v in pred and v != 0]
                tgt_tokens = [k for k, v in vocab.items() if v in tgt.tolist() and v != 0]
                pred_detok = sp.decode(pred_tokens)
                tgt_detok = sp.decode(tgt_tokens)
                hyps.append(pred_detok.split())
                refs.append([tgt_detok.split()])
    smoothie = SmoothingFunction().method4
    bleu = corpus_bleu(refs, hyps, smoothing_function=smoothie) * 100
    return bleu, refs, hyps

def main():
    config = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = spm.SentencePieceProcessor()
    sp.load("data/joint.model")
    vocab = load_vocab("data/joint.vocab")
    char_vocab = {c: i+1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?!-_:;'\"")}
    char_vocab["<unk>"] = 0
    model = SBCLTEncoderDecoder(
        vocab_size=len(vocab),
        char_vocab_size=len(char_vocab),
        config=config
    )
    checkpoint = torch.load("best_model.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    valid_data = TranslationDataset("data/tokenized/valid.kin.sp", "data/tokenized/valid.en.sp", vocab, char_vocab)
    valid_loader = DataLoader(valid_data, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch)
    bleu, refs, hyps = evaluate_bleu(model, valid_loader, vocab, char_vocab, config, device, sp)
    logging.info(f"\n🎯 Corpus BLEU (detokenized): {bleu:.2f}")
    # Print a few sample translations
    logging.info("\nSample translations:")
    for i in range(min(5, len(refs))):
        logging.info(f"Target:    {' '.join(refs[i][0])}")
        logging.info(f"Predicted: {' '.join(hyps[i])}\n")

if __name__ == "__main__":
    main()
