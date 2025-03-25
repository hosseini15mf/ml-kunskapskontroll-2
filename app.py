import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from functions import plot_image, plot_img_values2
from functions import *


scaler = joblib.load(get_scaler_name())
rf = joblib.load(get_rf_model_name())
svm = joblib.load(get_svm_model_name())
lr = joblib.load(get_lr_model_name())
best_model = joblib.load(get_best_model_name())

st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("Handwritten Digit Recognizer")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Draw a digit below:")
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=20,
        stroke_color="black",
        background_color="white",
        height=200,
        width=200,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    if canvas_result.image_data is not None:
        img = canvas_result.image_data
        img_rgba = Image.fromarray(img.astype('uint8'), 'RGBA')
        img_gray = img_rgba.convert('L')
        img_resized = img_gray.resize((28, 28))
        img_inverted = ImageOps.invert(img_resized)
        img_flattened = np.array(img_inverted).reshape(1, -1)

        plot_image(np.array(img_inverted).reshape(28, 28), "Processed Image")
        plot_img_values2(np.array(img_inverted).reshape(28, 28))

        img_scaled = scaler.transform(img_flattened)

        rf_pred = rf.predict(img_scaled)
        svm_pred = svm.predict(img_scaled)
        lr_pred = lr.predict(img_scaled)
        best_model_pred = best_model.predict(img_scaled)

        st.info(f"Random Forest Prediction: {rf_pred[0]}")
        st.info(f"SVM Prediction: {svm_pred[0]}")
        st.info(f"Logistic Regression Prediction: {lr.predict(img_scaled)[0]}")
        st.success(f"Best model Prediction: {best_model.predict(img_scaled)[0]}")

st.markdown("---")

st.subheader("Upload an image of a digit:")
uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L") 
    image = image.resize((28, 28))
    image = np.array(image)
    image = 255 - image
    
    image_flattened = image.flatten().reshape(1, -1)
    image_scaled = scaler.transform(image_flattened) 

    rf_pred_upload = rf.predict(image_scaled)
    svm_pred_upload = svm.predict(image_scaled)
    lr_pred_upload = lr.predict(image_scaled)
    best_model_pred_upload = best_model.predict(image_scaled)

    st.image(uploaded_file, caption="Uploaded Image", width=200)
    
    st.info(f"Random Forest Prediction: {rf_pred_upload[0]}")
    st.info(f"SVM Prediction: {svm_pred_upload[0]}")
    st.info(f"Logistic Regression Prediction: {lr_pred_upload[0]}")
    st.success(f"Best model Prediction: {best_model_pred_upload[0]}")
    
    st.info(f"The best model is {best_model}")

st.markdown("---")

