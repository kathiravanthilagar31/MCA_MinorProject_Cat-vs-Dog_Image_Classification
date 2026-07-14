import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

def get_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def superimpose_heatmap(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    
    # Convert BGR to RGB for correct Streamlit display
    superimposed_img = cv2.cvtColor(np.uint8(superimposed_img), cv2.COLOR_BGR2RGB)
    return superimposed_img

@st.cache_resource
def load_classification_model():
    return tf.keras.models.load_model('cat_dog_resnet50_finetuned.h5')

model = load_classification_model()

st.title("Cat vs Dog Classifier with Explainable AI")
st.write("Upload an image to classify and view the Grad-CAM heatmap.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Resize image to match model input requirements
    image_resized = image.resize((224, 224))
    img_array = img_to_array(image_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    predicted_class = "Dog" if prediction[0][0] > 0.5 else "Cat"
    st.write(f"**Prediction:** {predicted_class}")

    heatmap = get_gradcam_heatmap(img_array, model, 'conv5_block3_out')
    
    temp_img_path = "temp.jpg"
    image.save(temp_img_path)
    superimposed_img = superimpose_heatmap(temp_img_path, heatmap)
    
    st.image(superimposed_img, caption='Grad-CAM Heatmap', use_column_width=True)