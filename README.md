# 🧠 Image Deblurring using Deep Learning

This project restores sharpness to blurred images using a Convolutional Autoencoder trained on synthetic blur data.

---

## 🚀 Features

- Autoencoder architecture built with TensorFlow/Keras
- Synthetic Gaussian blur generation from clean images
- Visual evaluation of results (blurred vs deblurred)
- Command-line interface (CLI) for single-image testing
- Modular and extensible codebase (easy to swap model or blur type)

---

## 🗂️ Project Structure

image_deblurring_autoencoder/
├── dataset/
│ ├── sharp/ # Original clean images
│ └── blurred/
│ ├── generated_blur/ # Auto-generated blurred images
│ ├── motion_blurred/ # (Optional real blur)
│ └── defocused_blurred/ # (Optional real blur)
├── notebooks/
│ └── image_deblurring_dev.ipynb
├── src/
│ ├── prepare_data.py # Generate synthetic blurred images
│ ├── autoencoder_model.py # CNN autoencoder definition
│ ├── train.py # Model training
│ ├── evaluate.py # Visual prediction comparison
│ └── deblur_cli.py # CLI image deblurring tool
├── results/ # Output deblurred images
├── saved_model/
│ └── image_deblurring_autoencoder.h5
├── requirements.txt
└── README.md

---

## 📷 Sample Result

| Blurred | Deblurred | Sharp |
|--------|-----------|-------|
| ![Blur](assets/blur.jpg) | ![Deblur](assets/deblur.jpg) | ![Sharp](assets/sharp.jpg) |


---

## 🔧 How to Run

### 1. 📥 Install Requirements

- pip install -r requirements.txt

### 2. 🌀 Generate Blurred Images

- python src/prepare_data.py

3. 🏋️ Train the Autoencoder

- python src/train.py

4. 🧪 Evaluate on Test Set

- python src/evaluate.py

5. 🖼️ Deblur Any Image (CLI)

- python src/deblur_cli.py dataset/blurred/generated_blur/sample1.jpg

---

# 🧠 Model

- Architecture: Symmetric CNN autoencoder
- Loss Function: MSE
- Input Size: 64x64 RGB
- Training: 10 epochs (adjustable)

---

# 📝 License

- MIT License © 2025 Abhishek Thakare
