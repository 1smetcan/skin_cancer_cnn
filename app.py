import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

@st.cache_resource
def load_cancer_model():
    return load_model("cancer_cnn_model.h5")

model = load_cancer_model()

def process_image(img):
    img = img.resize((170, 170))
    img = np.array(img)
    img = img / 255
    img = np.expand_dims(img, axis=0)
    return img

st.title("Skin Cancer Prediction")
st.write("Select an image to check for signs of skin cancer.")

file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if file is not None:
    img = Image.open(file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    image = process_image(img)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
        
    class_names = ["Non-cancerous", "Cancerous"]
    st.markdown(f"<h2 style='text-align: center;'>{class_names[predicted_class]}</h2>", unsafe_allow_html=True)
