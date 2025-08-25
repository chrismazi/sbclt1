import sentencepiece as spm

spm.SentencePieceTrainer.Train(
    input='data/joint.txt',
    model_prefix='data/joint',
    vocab_size=32000,
    model_type='bpe',
    character_coverage=1.0,
    user_defined_symbols=['<s>', '</s>', '<unk>'],
    input_sentence_size=1000000,
    shuffle_input_sentence=True,
    train_extremely_large_corpus=True
)
