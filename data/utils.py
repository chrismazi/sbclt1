def load_vocab(vocab_file):
    vocab = {}
    with open(vocab_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            tok = line.strip().split("\t")[0]
            vocab[tok] = i
    return vocab

def decode_ids(token_ids, inv_vocab):
    return [inv_vocab.get(i, "<unk>") for i in token_ids]
