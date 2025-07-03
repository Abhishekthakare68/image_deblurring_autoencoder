import os
import cv2
import numpy as np

IMG_SIZE = (128, 128)
SHARP_PATH = r"C:\Users\abhis\Desktop\image_deblurring_autoencoder\dataset\sharp"
SAVE_PATH = r"C:\Users\abhis\Desktop\image_deblurring_autoencoder\dataset\blurred\generated_blur"

os.makedirs(SAVE_PATH, exist_ok=True)

def load_sharp_images(folder, max_images=500):
    images = []
    filenames = []
    for i, filename in enumerate(os.listdir(folder)):
        if i >= max_images:
            break
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder, filename)
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, IMG_SIZE)
                images.append(img)
                filenames.append(filename)
    return np.array(images), filenames

def apply_gaussian_blur(images):
    blurred = []
    for img in images:
        blur = cv2.GaussianBlur(img, (5, 5), 1)
        blurred.append(blur)
    return np.array(blurred)

def save_blurred_images(blurred, filenames):
    for img, name in zip(blurred, filenames):
        path = os.path.join(SAVE_PATH, name)
        cv2.imwrite(path, img)

if __name__ == "__main__":
    print("📥 Loading sharp images...")
    sharp_images, filenames = load_sharp_images(SHARP_PATH)
    print(f"✅ Loaded {len(sharp_images)} images.")

    print("🌀 Applying blur...")
    blurred_images = apply_gaussian_blur(sharp_images)

    print("💾 Saving blurred images...")
    save_blurred_images(blurred_images, filenames)

    print("✅ Done. Blurred images saved to:", SAVE_PATH)
