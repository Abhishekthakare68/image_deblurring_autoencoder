import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Config
MODEL_PATH = r"C:\Users\abhis\Desktop\image_deblurring_autoencoder\saved_model\image_deblurring_autoencoder.h5"
IMG_SIZE = (128, 128)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found:", image_path)
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    return img

def show_images(blurred, deblurred):
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plt.imshow(blurred)
    plt.title("Input (Blurred)")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(deblurred)
    plt.title("Output (Deblurred)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deblur an image using trained autoencoder")
    parser.add_argument("image_path", help="Path to the blurred image")
    args = parser.parse_args()

    print("📦 Loading model...")
    model = load_model(MODEL_PATH)

    print("📥 Loading and preprocessing image...")
    img = preprocess_image(args.image_path)
    input_img = np.expand_dims(img, axis=0)  # shape: (1, 64, 64, 3)

    print("🔮 Predicting...")
    output = model.predict(input_img)[0]

    print("🖼️ Showing result...")
    show_images(img, output)
