import base64
from io import BytesIO

from ddpm import guide, load_ddpm_pipeline
from flask import Flask, render_template, request
from PIL import Image

image_pipe, scheduler, device = load_ddpm_pipeline()

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


def open_image(file):
    """
    Opens an image file and handles PNG, JPG, JPEG, and HEIC formats.
    """
    try:
        file_type = file.content_type.lower()
        if file_type in ("image/heic", "image/heif"):
            import pyheif

            heif_file = pyheif.read(file.read())
            image = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw",
                heif_file.mode,
                heif_file.stride,
            )
        elif file_type in ("image/png", "image/jpeg", "image/jpg"):
            image = Image.open(file)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        return image
    except Exception as e:
        raise ValueError(f"Error opening image: {e}")


@app.route("/", methods=["GET", "POST"])
def image_gen():
    target_b64 = None
    generated_b64 = None

    if request.method == "POST":
        if "image" not in request.files:
            return "No file part"
        file = request.files["image"]
        if file.filename == "":
            return "No selected file"

        target_image = open_image(file)
        images = guide(image_pipe, scheduler, device, target_image)
        # images = [target_image] * 5
        target_b64 = images_to_base64(target_image)[0]

        generated_b64 = images_to_base64(images)
    return render_template("index.html", target=target_b64, generated=generated_b64)


if __name__ == "__main__":
    app.run(debug=True)