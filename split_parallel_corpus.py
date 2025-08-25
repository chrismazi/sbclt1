import random
import os

# Path to your downloaded TSV file
INPUT_FILE = r"C:\Users\user\Documents\GitHub\SBCLT\kinyarwanda-english-corpus.tsv"

# Output folder (will be created if it doesn't exist)
OUTPUT_DIR = r"C:\Users\user\Documents\GitHub\SBCLT\data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Proportions for splitting
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1

# Read and shuffle the data
with open(INPUT_FILE, "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")
random.shuffle(lines)

# Split lines into train/valid/test
total = len(lines)
train_end = int(train_ratio * total)
valid_end = train_end + int(valid_ratio * total)

train_lines = lines[:train_end]
valid_lines = lines[train_end:valid_end]
test_lines = lines[valid_end:]

# Helper function to write split files
def write_split(split_name, split_lines):
    with open(os.path.join(OUTPUT_DIR, f"{split_name}.en"), "w", encoding="utf-8") as f_en, \
         open(os.path.join(OUTPUT_DIR, f"{split_name}.kin"), "w", encoding="utf-8") as f_kin:
        for line in split_lines:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue  # Skip malformed lines
            kin, en = parts
            f_en.write(en.strip() + "\n")
            f_kin.write(kin.strip() + "\n")

# Write all splits
write_split("train", train_lines)
write_split("valid", valid_lines)
write_split("test", test_lines)

print("✅ Done! Files saved to 'data/' folder:")
print("- train.en / train.kin")
print("- valid.en / valid.kin")
print("- test.en / test.kin")
