import base64
from io import BytesIO

from ddpm import generate, load_ddpm_pipeline
from flask import Flask, render_template, request
from PIL import Image

# image_pipe, scheduler, device = load_ddpm_pipeline()
# generate(image_pipe, scheduler, device)

app = Flask(__name__)


def images_to_base64(images, format="JPEG"):
    if not isinstance(images, list):
        images = [images]
    b64_images = []
    for img in images:
        img_io = BytesIO()
        img.save(img_io, format)
        img_io.seek(0)
        b64_images.append(base64.b64encode(img_io.read()).decode("utf-8"))
    return b64_images


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
        images = [img] * 5

        # Convert original image to base64
        original_b64 = images_to_base64(img)[0]

        # Resize the image (e.g., max 300x300)
        img.thumbnail((300, 300))
        resized_b64 = images_to_base64(images)
    return render_template("index.html", original=original_b64, resized=resized_b64)


if __name__ == "__main__":
    app.run(debug=True)
