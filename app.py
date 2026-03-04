from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import pickle
import os

app = Flask(__name__)

DATABASE = "faces.pkl"

# Load face database
if os.path.exists(DATABASE):
    with open(DATABASE, "rb") as f:
        data = pickle.load(f)
else:
    data = {"encodings": [], "names": []}


@app.route("/")
def home():
    return "Face Recognition API Running"


# ---------------- REGISTER FACE ----------------
@app.route("/register", methods=["POST"])
def register():

    name = request.form.get("name")

    if "image" not in request.files:
        return jsonify({"status": "error", "message": "no image"})

    image = request.files["image"]

    img = face_recognition.load_image_file(image)
    encodings = face_recognition.face_encodings(img)

    if len(encodings) == 0:
        return jsonify({"status": "error", "message": "no face detected"})

    encoding = encodings[0]

    data["encodings"].append(encoding)
    data["names"].append(name)

    with open(DATABASE, "wb") as f:
        pickle.dump(data, f)

    return jsonify({
        "status": "registered",
        "name": name
    })


# ---------------- RECOGNIZE FACE ----------------
@app.route("/recognize", methods=["POST"])
def recognize():

    if "image" not in request.files:
        return jsonify({"recognized": False})

    image = request.files["image"]

    img = face_recognition.load_image_file(image)
    encodings = face_recognition.face_encodings(img)

    if len(encodings) == 0:
        return jsonify({"recognized": False})

    face = encodings[0]

    distances = face_recognition.face_distance(data["encodings"], face)

    if len(distances) == 0:
        return jsonify({"recognized": False})

    best_match = np.argmin(distances)
    probability = 1 - distances[best_match]

    if probability > 0.6:

        return jsonify({
            "recognized": True,
            "name": data["names"][best_match],
            "probability": float(probability)
        })

    return jsonify({"recognized": False})


# ---------------- LIST REGISTERED USERS ----------------
@app.route("/faces", methods=["GET"])
def list_faces():

    return jsonify({
        "faces": data["names"]
    })


# ---------------- DELETE FACE ----------------
@app.route("/delete", methods=["POST"])
def delete_face():

    name = request.form.get("name")

    if name not in data["names"]:
        return jsonify({"status": "not found"})

    index = data["names"].index(name)

    data["names"].pop(index)
    data["encodings"].pop(index)

    with open(DATABASE, "wb") as f:
        pickle.dump(data, f)

    return jsonify({"status": "deleted", "name": name})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
