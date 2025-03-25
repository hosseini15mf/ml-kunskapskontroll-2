import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import streamlit as st
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageOps

svm_model = "svm_model.pkl"
rf_model = "rf_model.pkl"
lr_model = "lr_model.pkl"
best_model = "best_model.pkl"
scaler = "scaler.pkl"

def get_svm_model_name():
    return svm_model

def get_rf_model_name():
    return rf_model

def get_lr_model_name():
    return lr_model

def get_best_model_name():
    return best_model

def get_scaler_name():
    return scaler
    

def plot_images(images, labels, nrows, ncols):
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10))
    axes = axes.flatten()
    for img, ax, label in zip(images, axes, labels):
        ax.imshow(img.reshape(28, 28), cmap='gray')
        ax.set_title(f'Label: {label}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    
def plot_image(img_2d, label):
    st.write("shape: ", img_2d.shape)
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.imshow(img_2d, cmap='gray')
    st.pyplot(fig, use_container_width=False)
    
    
def plot_img_values(img_2d):
    st.write("Pixel values of the first MNIST image (28x28 grid):")
    st.markdown("""
    <style>
    pre {
        font-family: 'Courier New', monospace;
        font-size: 6px;
        white-space: pre-wrap;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<pre>", unsafe_allow_html=True)
    for row in img_2d:
        row_str = "  ".join(map(lambda x: f"{x:6}", row))
        st.markdown(row_str)
    st.markdown("</pre>", unsafe_allow_html=True)
    
    
def plot_img_values2(img_2d):
    fig, ax = plt.subplots()
    ax.matshow(img_2d, cmap='gray')

    for i in range(img_2d.shape[0]):
        for j in range(img_2d.shape[1]):
            c = img_2d[i, j]
            ax.text(j, i, str(c), va='center', ha='center', color='red')

    plt.show()
    

def get_score(model, x_val, y_val):
    y_pred = model.predict(x_val)
    score = accuracy_score(y_val, y_pred)
    print(f"{model}, accuracy score: {score} ")
    return score
        
        
        
    
    