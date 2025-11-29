from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import google.generativeai as genai
import requests
import re

app = Flask(__name__)

# ---- Configure Google Gemini ----
genai.configure(api_key="AIzaSyAJl6zuh_8VyyYcOg0b1UeYsuzUZr6aIjM")  # replace with your key
model_gemini = genai.GenerativeModel("gemini-1.5-flash")

# ---- Load Model ----
MODEL_PATH = "dog_breed_mobilenetv2_finetuned.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# ---- Load class names ----
import json
with open("label_map.json", "r") as f:
    breed_names = json.load(f)

# ---- Helper to clean breed name ----
def clean_breed_name(raw_name):
    return re.sub(r"n\d+-", "", raw_name).replace("_", " ").strip().title()

# ---- Image preprocessing ----
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# ---- Breed prediction ----
@app.route("/", methods=["GET", "POST"])
def index():
    breed = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            path = "static/" + file.filename
            file.save(path)
            img = preprocess_image(path)
            pred = model.predict(img)
            idx = np.argmax(pred)
            breed = clean_breed_name(breed_names[idx])
            return render_template("index.html", breed=breed, image_path=path)
    return render_template("index.html", breed=breed)

# ---- Gemini + Wikipedia Chatbot ----
def get_breed_info(query, breed):
    try:
        # Try Gemini first
        prompt = f"You are a dog expert. Answer briefly: {query} about the dog breed {breed}."
        response = model_gemini.generate_content(prompt)
        if response.text and "error" not in response.text.lower():
            return response.text.strip()
    except Exception:
        pass  # Fall back to Wikipedia if Gemini fails

    # Fallback: Wikipedia API
    try:
        wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{breed.replace(' ', '_')}"
        res = requests.get(wiki_url).json()
        if "extract" in res:
            return res["extract"]
    except Exception:
        pass

    return f"Sorry, I couldn’t find any info about {breed}."

# ---- Chat endpoint ----
@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json["message"]
    breed = request.json["breed"]
    bot_reply = get_breed_info(user_msg, breed)
    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)
