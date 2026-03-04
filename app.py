from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

DATASET = "dataset"
MODEL = "model.yml"

if not os.path.exists(DATASET):
    os.makedirs(DATASET)

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

names = []


def train_model():
    faces = []
    labels = []
    names.clear()

    for idx, person in enumerate(os.listdir(DATASET)):
        person_path = os.path.join(DATASET, person)
        names.append(person)

        for img_name in os.listdir(person_path):
            path = os.path.join(person_path, img_name)
            img = Image.open(path).convert("L")
            img_np = np.array(img, "uint8")

            faces.append(img_np)
            labels.append(idx)

    if len(faces) > 0:
        recognizer.train(faces, np.array(labels))
        recognizer.save(MODEL)


if os.path.exists(MODEL):
    recognizer.read(MODEL)


@app.route("/")
def home():
    return "Face API running"


@app.route("/register", methods=["POST"])
def register():

    name = request.form.get("name")

    if "image" not in request.files:
        return jsonify({"status": "no image"})

    file = request.files["image"]

    person_dir = os.path.join(DATASET, name)

    if not os.path.exists(person_dir):
        os.makedirs(person_dir)

    img = Image.open(file).convert("L")
    img_np = np.array(img)

    faces = face_cascade.detectMultiScale(img_np, 1.3, 5)

    if len(faces) == 0:
        return jsonify({"status": "no face detected"})

    for (x, y, w, h) in faces:
        face = img_np[y:y+h, x:x+w]
        cv2.imwrite(f"{person_dir}/{len(os.listdir(person_dir))+1}.jpg", face)

    train_model()

    return jsonify({"status": "registered", "name": name})


@app.route("/recognize", methods=["POST"])
def recognize():

    if "image" not in request.files:
        return jsonify({"recognized": False})

    file = request.files["image"]

    img = Image.open(file).convert("L")
    img_np = np.array(img)

    faces = face_cascade.detectMultiScale(img_np, 1.3, 5)

    for (x, y, w, h) in faces:
        face = img_np[y:y+h, x:x+w]

        label, confidence = recognizer.predict(face)

        probability = max(0, 100 - confidence)

        if label < len(names):

            return jsonify({
                "recognized": True,
                "name": names[label],
                "probability": probability
            })

    return jsonify({"recognized": False})


@app.route("/faces")
def list_faces():
    return jsonify({"faces": os.listdir(DATASET)})


if __name__ == "__main__":
    train_model()
    app.run(host="0.0.0.0", port=10000)
