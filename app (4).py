import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import matplotlib.pyplot as plt

st.title("DCGAN Synthetic Photo Generator")

import tensorflow as tf
import streamlit as st

st.write("TensorFlow Version:", tf.__version__)

# Safely load the dataset without crashing if directory is missing
def load_dataset_safe(directory='MyPhotoDataset', img_size=(64, 64)):
    images = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    if os.path.exists(directory):
        filenames = [f for f in os.listdir(directory) if f.lower().endswith(valid_extensions)]
        for filename in filenames:
            try:
                img_path = os.path.join(directory, filename)
                img = Image.open(img_path).resize(img_size)
                img_array = img_to_array(img)
                img_array = (img_array - 127.5) / 127.5
                images.append(img_array)
            except Exception as e:
                st.warning(f"Error processing {filename}: {e}")
    else:
        st.info(f"Local directory '{directory}' not found in deployment runtime. You can upload custom files below to test.")
        uploaded_files = st.file_uploader("Upload real images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    img = Image.open(uploaded_file).resize(img_size)
                    img_array = img_to_array(img)
                    img_array = (img_array - 127.5) / 127.5
                    images.append(img_array)
                except Exception as e:
                    st.error(f"Error loading uploaded file: {e}")
    return np.array(images)

# Load the models safely
@st.cache_resource
def load_generator():
    model_path = 'saved_models/dcgan_generator.keras'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        st.error(f"Model file not found at {model_path}. Please upload or place it in your repository.")
        return None

generator = load_generator()

if generator is not None:
    if st.button("Generate Synthetic Photos"):
        latent_dim = 100
        noise = tf.random.normal([4, latent_dim])
        predictions = generator(noise, training=False)
        predictions = (predictions + 1.0) / 2.0  # Rescale to [0, 1]
        
        fig, axes = plt.subplots(1, 4, figsize=(10, 3))
        for i in range(4):
            axes[i].imshow(predictions[i].numpy())
            axes[i].axis('off')
        st.pyplot(fig)
