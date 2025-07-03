import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Paths
MODEL_PATH = r"C:\Users\abhis\Desktop\image_deblurring_autoencoder\saved_model\image_deblurring_autoencoder.h5"
SHARP_DIR = r"C:\Users\abhis\Desktop\image_deblurring_autoencoder\dataset\sharp"
BLUR_DIR = r"C:\Users\abhis\Desktop\image_deblurring_autoencoder\dataset\blurred\generated_blur"

IMG_SIZE = (128, 128)
NUM_SAMPLES = 5

def load_images(folder, filenames):
    images = []
    for name in filenames:
        path = os.path.join(folder, name)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, IMG_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    return np.array(images).astype("float32") / 255.0

def show_predictions(blurred, predicted, sharp):
    n = len(blurred)
    plt.figure(figsize=(12, 6))
    for i in range(n):
        # Blurred input
        plt.subplot(3, n, i+1)
        plt.imshow(blurred[i])
        plt.title("Blurred")
        plt.axis("off")

        # Deblurred prediction
        plt.subplot(3, n, i+1+n)
        plt.imshow(predicted[i])
        plt.title("Deblurred")
        plt.axis("off")

        # Original sharp
        plt.subplot(3, n, i+1+2*n)
        plt.imshow(sharp[i])
        plt.title("Sharp")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("📦 Loading model...")
    model = load_model(MODEL_PATH)

    print("📥 Loading sample images...")
    all_filenames = os.listdir(BLUR_DIR)
    sample_names = [name for name in all_filenames if name.lower().endswith(('.jpg', '.jpeg', '.png'))][:NUM_SAMPLES]

    X_blur = load_images(BLUR_DIR, sample_names)
    y_true = load_images(SHARP_DIR, sample_names)

    print("🔮 Running predictions...")
    y_pred = model.predict(X_blur)

    print("🖼️ Showing comparisons...")
    show_predictions(X_blur, y_pred, y_true)
