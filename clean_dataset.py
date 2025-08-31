# clean_dataset.py

import pandas as pd

input_file = "data/SMSSpamCollection"
output_file = "data/SMSSpamCollection_cleaned.csv"

print("📥 Reading raw dataset...")
rows = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 2:  # valid row has exactly 2 parts
            rows.append(parts)

print(f"✅ Total valid rows: {len(rows)}")

# Save cleaned dataset
df = pd.DataFrame(rows, columns=["label", "message"])
df.to_csv(output_file, index=False)

print(f"🎉 Cleaned dataset saved to {output_file}")
print(df.head())
