from flask import Flask, render_template, request, jsonify
import whisper
from transformers import pipeline
import os

app = Flask(__name__)

# Load Whisper model
model = whisper.load_model("base")

# Load Sentiment model
sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["audio"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        # Speech to text
        result = model.transcribe(filepath)
        text = result["text"]

        # Sentiment analysis
        sentiment_result = sentiment(text)

        return jsonify({
            "transcription": text,
            "sentiment": sentiment_result
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run()