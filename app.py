from flask import Flask, request, jsonify, render_template, redirect, url_for
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import uuid
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score  

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = load_model("mobilenet_model.h5")

class_names = [
    "battery", "biological", "brown-glass", "cardboard",
    "clothes", "green-glass", "metal", "paper",
    "plastic", "shoes", "trash", "white-glass"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/map")
def map_page():
    return render_template("map.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    filename = str(uuid.uuid4()) + ".jpg"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    img = Image.open(filepath).resize((224, 224)).convert("RGB")
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    prediction = model.predict(img_array)
    class_id = int(np.argmax(prediction))

   
    y_true = np.zeros(len(class_names)) 
    y_true[class_id] = 1 
    
    
    y_pred = prediction[0]  
    confidence = float(np.max(prediction))

    return render_template("index.html", 
        prediction=class_names[class_id],
        confidence=confidence,
        image_url=url_for('static', filename='uploads/' + filename),
        all_probs=prediction.tolist(),
        class_names=class_names  
    )

if __name__ == "__main__":
    app.run(debug=True)
