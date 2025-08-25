import torch
from torch.utils.data import DataLoader, Dataset
from model.sbclt_seq2seq import SBCLTEncoderDecoder
from utils import load_vocab
import sentencepiece as spm
from config import ModelConfig
from model.beam_search import beam_search
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

class MonolingualDataset(Dataset):
    def __init__(self, mono_file, sp_model):
        self.lines = open(mono_file, encoding="utf-8").read().splitlines()
        self.sp = sp_model
    def __len__(self):
        return len(self.lines)
    def __getitem__(self, idx):
        tokens = self.sp.encode(self.lines[idx], out_type=str)
        return tokens

def collate_batch(batch, vocab, char_vocab, max_char_len):
    max_len = max(len(x) for x in batch)
    ids = []
    chars = []
    for tokens in batch:
        id_seq = [vocab.get(tok, vocab['<unk>']) for tok in tokens]
        char_seq = [[char_vocab.get(c, 0) for c in list(tok)[:max_char_len]] + [0]*(max_char_len-len(list(tok)[:max_char_len])) for tok in tokens]
        # Pad
        id_seq += [0] * (max_len - len(id_seq))
        char_seq += [[0]*max_char_len] * (max_len - len(char_seq))
        ids.append(id_seq)
        chars.append(char_seq)
    return torch.tensor(ids), torch.tensor(chars)

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
    # Load monolingual English data
    mono_file = "data/monolingual.en"  # <-- You need to provide this file
    dataset = MonolingualDataset(mono_file, sp)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    out_src = open("data/backtrans.kin", "w", encoding="utf-8")
    out_tgt = open("data/backtrans.en", "w", encoding="utf-8")
    for batch in loader:
        ids, chars = collate_batch(batch, vocab, char_vocab, config.max_char_len)
        ids, chars = ids.to(device), chars.to(device)
        for i in range(ids.size(0)):
            pred = beam_search(
                model=model,
                src_ids=ids[i:i+1],
                src_chars=chars[i:i+1],
                vocab=vocab,
                char_vocab=char_vocab,
                config=config,
                device=device
            )
            pred_tokens = [k for k, v in vocab.items() if v in pred and v != 0]
            pred_detok = sp.decode(pred_tokens)
            src_detok = sp.decode(batch[i])
            out_src.write(pred_detok + "\n")
            out_tgt.write(src_detok + "\n")
    out_src.close()
    out_tgt.close()
    logging.info("Back-translation complete. Synthetic data written to data/backtrans.kin and data/backtrans.en")

if __name__ == "__main__":
    main() 