from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from numpy.linalg import norm

app = Flask(__name__)

FACE_DIR = "faces"

face_app = FaceAnalysis()
face_app.prepare(ctx_id=0)

def cosine(a,b):
    return np.dot(a,b)/(norm(a)*norm(b))


@app.route("/")
def index():
    return render_template("register.html")


# REGISTER FACE

@app.route("/register", methods=["POST"])
def register():

    name = request.form["name"]
    file = request.files["image"]

    path = os.path.join(FACE_DIR, name+".jpg")

    file.save(path)

    return "saved"


# RECOGNIZE FACE

@app.route("/recognize", methods=["POST"])
def recognize():

    file = request.files["image"]

    img = cv2.imdecode(np.frombuffer(file.read(),np.uint8),cv2.IMREAD_COLOR)

    faces = face_app.get(img)

    if len(faces)==0:
        return jsonify({"result":"unknown"})

    emb = faces[0].embedding

    best = 0
    best_name = "unknown"

    for f in os.listdir(FACE_DIR):

        db_img = cv2.imread(os.path.join(FACE_DIR,f))
        db_face = face_app.get(db_img)

        if len(db_face)==0:
            continue

        score = cosine(emb, db_face[0].embedding)

        if score > best:
            best = score
            best_name = f

    if best > 0.4:
        return jsonify({"result":"recognized","name":best_name})

    return jsonify({"result":"unknown"})


if __name__ == "__main__":
    app.run()
