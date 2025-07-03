import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from autoencoder_model import build_autoencoder
from tensorflow.keras.callbacks import EarlyStopping

# Configuration
IMG_SIZE = (128, 128)
SHARP_DIR = r"C:\Users\abhis\Desktop\image_deblurring_autoencoder\dataset\sharp"
BLUR_DIR = r"C:\Users\abhis\Desktop\image_deblurring_autoencoder\dataset\blurred\generated_blur"
MODEL_PATH = r"C:\Users\abhis\Desktop\image_deblurring_autoencoder\saved_model\image_deblurring_autoencoder.h5"
EPOCHS = 50
BATCH_SIZE = 32

def load_images(folder, max_images=500):
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
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                filenames.append(filename)
    return np.array(images).astype("float32") / 255.0, filenames

if __name__ == "__main__":
    print("📥 Loading blurred and sharp images...")
    X_blur, filenames = load_images(BLUR_DIR)
    y_sharp, _ = load_images(SHARP_DIR)

    print(f"✅ Loaded {len(X_blur)} image pairs.")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_blur, y_sharp, test_size=0.2, random_state=42)

    # Build model
    model = build_autoencoder(input_shape=(128, 128, 3))
    model.summary()

    # Train model
    print("🚀 Starting training...")
    checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss', mode='min')
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint]
    )

    print("✅ Training complete. Model saved to:", MODEL_PATH)



