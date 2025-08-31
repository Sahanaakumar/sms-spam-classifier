import joblib

# Load model & vectorizer
MODEL_PATH = "models/spam_classifier.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Test samples
messages = [
    "Congratulations! You've won a $1000 Walmart gift card. Call now!",
    "Hey, are we still meeting for dinner tonight?",
    "URGENT! Your account has been compromised. Click here to secure it."
]

# Transform input & predict
X_test = vectorizer.transform(messages)
predictions = model.predict(X_test)

for msg, pred in zip(messages, predictions):
    label = "SPAM" if pred == 1 else "HAM"
    print(f"Message: {msg}\nPrediction: {label}\n")
