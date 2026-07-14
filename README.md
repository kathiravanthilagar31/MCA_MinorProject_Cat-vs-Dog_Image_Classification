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

```

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/kathiravanthilagar31/MCA_MinorProject_Cat-vs-Dog_Image_Classification.git](https://github.com/kathiravanthilagar31/MCA_MinorProject_Cat-vs-Dog_Image_Classification.git)
cd MCA_MinorProject_Cat-vs-Dog_Image_Classification

```

### 2. Set Up Virtual Environment (Optional but Recommended)

If you are using Conda:

```bash
conda create -n xai_project python=3.10 -y
conda activate xai_project

```

Or using standard Python venv:

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

---

## Running the Application

To launch the interactive web dashboard locally, execute the following command in your terminal:

```bash
streamlit run app.py

```

This will automatically open the web application in your default browser at `http://localhost:8501`.

---

## Model Training (Optional)

If you wish to retrain the ResNet50 model from scratch or test different hyperparameters, run:

```bash
python train.py

```

*Note: Training requires the Kaggle Dogs vs. Cats dataset formatted into standard `/train` and `/test` directory structures as defined inside `train.py`.*

---

## Key Technologies Used

* **Deep Learning Framework:** TensorFlow & Keras
* **Base Architecture:** ResNet50 (pre-trained on ImageNet)
* **Explainable AI:** Grad-CAM
* **Image Processing:** OpenCV (`cv2`) & Pillow (`PIL`)
* **Frontend Web Application:** Streamlit

```
