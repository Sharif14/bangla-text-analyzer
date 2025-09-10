import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re

# =============================
# মডেল লোড করা
# =============================
pkl_model = joblib.load("model/tokenizer.pkl")
h5_model = load_model("model/cnn_model.h5")

# =============================
# Tokenizer লোড + Max sequence length সেট করা
# =============================
tokenizer = joblib.load("model/tokenizer.pkl")
max_sequence_length = 100  # ট্রেইনিংয়ের সময়কার মান ব্যবহার করতে হবে

# =============================
# প্রিপ্রসেস ফাংশন
# =============================
def preprocess_text(text: str):
    text = re.sub(r"[^অ-ঔঅ-হa-zA-Z0-9\s]", "", text)
    text = text.strip().lower()
    return text

# =============================
# predict_text (H5 model দিয়ে)
# =============================
def predict_text(input_text):
    # tokenization & padding
    seq = tokenizer.texts_to_sequences([input_text])
    padded = pad_sequences(seq, maxlen=max_sequence_length)

    # prediction
    prediction = h5_model.predict(padded)
    predicted_label = 'সাইবারবুলিং' if prediction[0][0] > 0.5 else 'নন-সাইবারবুলিং'

    print(f"ইনপুট: {input_text}")
    print(f"প্রেডিকশন: {predicted_label} (প্রবাবিলিটি: {prediction[0][0]:.4f})")

# =============================
# predict_with_pkl
# =============================
def predict_with_pkl(text: str):
    processed = preprocess_text(text)
    prediction = pkl_model.predict([processed])[0]
    return prediction

# =============================
# predict_with_h5
# =============================
def predict_with_h5(text: str):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_sequence_length)
    prediction = h5_model.predict(padded)
    predicted_class = prediction.argmax(axis=1)[0]
    return predicted_class

def predict(text: str):
    # টেক্সট প্রিপ্রসেস (যদি দরকার হয়)
    processed = preprocess_text(text)

    # টোকেনাইজ + প্যাডিং
    seq = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(seq, maxlen=max_sequence_length)

    # প্রেডিকশন
    prediction = h5_model.predict(padded)

    prob = float(prediction[0][0])   # 0 থেকে 1 এর মধ্যে probability
    label = "সাইবারবুলিং" if prob > 0.5 else "নন-সাইবারবুলিং"

    return label, prob
