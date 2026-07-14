# Cat vs Dog Image Classification using Transfer Learning and Explainable AI (XAI)

This repository contains the end-to-end implementation of a deep learning image classification system designed to distinguish between cats and dogs. The project leverages **ResNet50** transfer learning for high-accuracy binary classification and integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)** to provide Explainable AI (XAI) visual verification.

An interactive web interface built with **Streamlit** allows users to upload raw images, view real-time prediction confidences, and inspect superimposed heatmaps that highlight the exact visual features driving the model's decisions.

---

## System Architecture

The application operates via a dual-path processing pipeline:
1. **Classification Branch:** The input image is preprocessed and passed through a fine-tuned ResNet50 model (with the `conv5` block unfreezed) to compute the probability score.
2. **Explainable AI Branch:** The gradient maps from the final convolutional layer (`conv5_block3_out`) are extracted using Grad-CAM, processed via OpenCV, and rendered as a red-blue JET colormap overlay.
3. **Interactive UI:** Streamlit serves both the classification result and the XAI heatmap simultaneously to complete the user feedback loop.

---

## Project Structure

```text
├── train.py           # Model training, augmentation, and ResNet50 fine-tuning script
├── app.py             # Streamlit web interface and Grad-CAM backend implementation
├── requirements.txt   # Required Python package dependencies
├── README.md          # Project documentation
└── *.h5               # Fine-tuned ResNet50 Keras model weights (tracked via Git LFS)
