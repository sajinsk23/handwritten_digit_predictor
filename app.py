from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

# Load trained model
model = load_model("digit_model.h5")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":

        img = request.files["file"]
        img = Image.open(img).convert("L").resize((28, 28))
        
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28)

        prediction = np.argmax(model.predict(img_array))

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)