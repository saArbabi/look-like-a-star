import base64
from io import BytesIO

from flask import Flask, render_template, request
from PIL import Image

from ddpm import generate, load_ddpm_pipeline

# image_pipe, scheduler, device = load_ddpm_pipeline()
# generate(image_pipe, scheduler, device)

app = Flask(__name__)


def image_to_base64(img, format="JPEG"):
    img_io = BytesIO()
    img.save(img_io, format)
    img_io.seek(0)
    return base64.b64encode(img_io.read()).decode("utf-8")


@app.route("/", methods=["GET", "POST"])
def upload_image():
    original_b64 = None
    resized_b64 = None

    if request.method == "POST":
        if "image" not in request.files:
            return "No file part"
        file = request.files["image"]
        if file.filename == "":
            return "No selected file"

        img = Image.open(file)

        # Convert original image to base64
        original_b64 = image_to_base64(img)

        # Resize the image (e.g., max 300x300)
        img.thumbnail((300, 300))
        resized_b64 = image_to_base64(img)

    return render_template("index.html", original=original_b64, resized=resized_b64)


if __name__ == "__main__":
    app.run(debug=True)
