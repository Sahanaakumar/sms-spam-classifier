from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load model and vectorizer
MODEL_PATH = os.path.join("models", "spam_classifier.pkl")
VECTORIZER_PATH = os.path.join("models", "vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]
    data = vectorizer.transform([message])

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0]

    # Map numeric label to text
    label = "Spam" if prediction == 1 or prediction == "spam" else "Ham"

    # Confidence (take probability of predicted class)
    confidence = round(max(probability) * 100, 2)

    return render_template(
        "index.html",
        prediction=label,
        confidence=confidence,
        message=message
    )

if __name__ == "__main__":
    app.run(debug=True)
