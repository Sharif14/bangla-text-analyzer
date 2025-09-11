# app.py
from __future__ import annotations
from flask import Flask, render_template, request, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix
from model.model import predict


app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)


@app.get("/")
def index():
    return render_template("index.html")

@app.post("/predict")
def predict_route():
    text = (request.form.get("text") or "").strip()
    if not text:
        return render_template(
            "index.html",
            error="অনুগ্রহ করে কিছু লেখা দিন।",
            last_text=text
        ), 400

    try:
        label, prob = predict(text)  # <- আপনার মডেল
    except Exception as e:
        return render_template(
            "index.html",
            error=f"প্রেডিকশনে সমস্যা হয়েছে: {e}",
            last_text=text
        ), 500

    # শতাংশে দেখানোর জন্য:
    prob_pct = round(prob * 100, 2)
    return render_template(
        "index.html",
        result={"label": label, "prob": prob, "prob_pct": prob_pct, "text": text},
        last_text=text
    )

# ঐচ্ছিক: JSON API এন্ডপয়েন্ট (POST: { "text": "..." })
@app.post("/api/predict")
def api_predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text ফিল্ড খালি"}), 400
    try:
        label, prob = predict(text)
    except Exception as e:
        return jsonify({"error": f"প্রেডিকশনে সমস্যা: {str(e)}"}), 500
    return jsonify({"label": label, "probability": float(prob)}), 200

if __name__ == "__main__":
    # লোকাল রান
    app.run(host="127.0.0.1", port=5000, debug=True)
