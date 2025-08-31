# src/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib

# --- Step 1: Load dataset ---
print("📥 Loading dataset...")
try:
   df = pd.read_csv("data/SMSSpamCollection_cleaned.csv")

except Exception as e:
    print("❌ Error loading dataset:", e)
    exit()

print("✅ First 5 rows of dataset:")
print(df.head())
print("📊 Dataset shape:", df.shape)

# --- Step 2: Clean and preprocess ---
print("🧹 Cleaning data...")
df.dropna(subset=["label", "message"], inplace=True)
df["message"] = df["message"].astype(str)

# Normalize labels
df["label"] = df["label"].str.strip().str.lower().map({"ham": 0, "spam": 1})

# Drop rows with unmapped labels (NaN)
df = df.dropna(subset=["label"])

print("🔎 Unique labels after mapping:", df["label"].unique())
print("📊 Dataset shape after cleaning:", df.shape)
print("📊 Label counts:\n", df["label"].value_counts())

# If dataset is still empty → exit
if df.shape[0] == 0:
    print("❌ Dataset is empty after cleaning. Check your file format/labels.")
    exit()

# --- Step 3: Train/test split ---
print("✂️ Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

print("📊 Train size:", X_train.shape[0], " Test size:", X_test.shape[0])

# --- Step 4: Vectorization ---
print("🔡 Vectorizing text...")
vectorizer = TfidfVectorizer(stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# --- Step 5: Train Model ---
print("🤖 Training model...")
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# --- Step 6: Evaluate ---
print("📈 Evaluating model...")
y_pred = model.predict(X_test_tfidf)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📋 Classification Report:\n", classification_report(y_test, y_pred))

# --- Step 7: Save model ---
print("💾 Saving model and vectorizer...")
joblib.dump(model, "models/spam_classifier.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
print("🎉 Training complete. Model saved!")
