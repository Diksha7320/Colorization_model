import numpy as np
import cv2
import gradio as gr

# File paths for the pre-trained model and data
PROTOTXT = "colorization_deploy_v2.prototxt"
POINTS = "pts_in_hull.npy"
MODEL = "colorization_release_v2.caffemodel"

# Load the model
print("Loading model...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

# Set model layers
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Colorization function
def colorize_image(image):
    if image is None:
        return None

    # Gradio gives RGB, OpenCV wants BGR
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w = image_bgr.shape[:2]

    # Convert to float and scale to [0, 1]
    img_rgb = image_bgr.astype("float32") / 255.0

    # Convert to Lab color space
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2Lab)
    L_channel = img_lab[:, :, 0]  # Only L channel

    # Resize to 224x224 before feeding into the model
    L_rs = cv2.resize(L_channel, (224, 224)) - 50  # subtract 50 mean-centering
    net.setInput(cv2.dnn.blobFromImage(L_rs))

    # Predict a & b channels
    ab_decoded = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_decoded_us = cv2.resize(ab_decoded, (w, h))

    # Concatenate with original L channel
    L_channel = L_channel[:, :, np.newaxis]
    colorized_lab = np.concatenate((L_channel, ab_decoded_us), axis=2)

    # Convert back to BGR
    colorized_bgr = cv2.cvtColor(colorized_lab, cv2.COLOR_Lab2BGR)
    colorized_bgr = np.clip(colorized_bgr, 0, 1)

    # Convert to uint8
    colorized_bgr = (colorized_bgr * 255).astype("uint8")

    # Convert back to RGB for Gradio
    return cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2RGB)


# Interface with side-by-side input-output display
demo = gr.Interface(
    fn=colorize_image,
    inputs=gr.Image(label="Input (Black & White)"),
    outputs=gr.Image(label="Output (Colorized)"),
    examples=[
        ["einstein.jpg"],
        ["tiger.jpg"],
        ["building.jpg"],
        ["nature.jpg"]
    ],
    title="Black & White to Color Image"
)

# Launch the UI
demo.launch(debug=True)
