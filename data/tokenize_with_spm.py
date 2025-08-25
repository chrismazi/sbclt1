import sentencepiece as spm
import os

# Paths
MODEL_PATH = "data/Models/joint.model"
FILES_TO_TOKENIZE = [
    "train.kin", "train.en",
    "valid.kin", "valid.en",
    "test.kin", "test.en",
]
INPUT_DIR = "data"
OUTPUT_DIR = "data/tokenized"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the trained tokenizer model
sp = spm.SentencePieceProcessor()
sp.load(MODEL_PATH)

# Tokenize each file
for filename in FILES_TO_TOKENIZE:
    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename + ".sp")

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            tokens = sp.encode(line, out_type=str)
            fout.write(" ".join(tokens) + "\n")

    print(f"✅ Tokenized: {filename} → {output_path}")
