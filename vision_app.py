import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# --- 1. Load and Prepare the Data ---
@st.cache_data
def load_data():
    """Loads and preprocesses the MNIST dataset."""
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    # Normalize images to be between 0 and 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    return train_images, train_labels, test_images, test_labels

# --- 2. Build and Train the AI Model ---
@st.cache_resource
def build_and_train_model(train_images, train_labels):
    """Builds and trains a simple neural network."""
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5)
    return model

# --- 3. Set Up the Streamlit App Interface ---
st.title("See Like an AI: Handwritten Digit Recognizer")
st.write("Draw a single digit (0-9) in the black box below and the AI will try to guess it.")

# Create a drawing canvas
canvas_result = st_canvas(
    stroke_width=15,
    stroke_color="#FFFFFF",  # White color to draw on a black background
    background_color="#000000",
    height=200,
    width=200,
    drawing_mode="freedraw",
    key="canvas",
)

# --- 4. Process the Drawing and Make a Prediction ---
if canvas_result.image_data is not None:
    # Convert the canvas data to a NumPy array
    img = np.array(canvas_result.image_data)
    # Convert RGBA to grayscale
    img_gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
    
    # Resize the image to 28x28 pixels to match the model's input size
    img_pil = Image.fromarray(img_gray.astype('uint8'))
    img_resized = img_pil.resize((28, 28), Image.LANCZOS)
    
    # Final preprocessing: normalize and reshape
    img_final = np.array(img_resized) / 255.0
    img_final = img_final.reshape(1, 28, 28)

    # Load data and model (Streamlit caches these so they run only once)
    train_images, train_labels, _, _ = load_data()
    model = build_and_train_model(train_images, train_labels)

    # Make the prediction!
    prediction = model.predict(img_final)
    predicted_class = np.argmax(prediction)

    st.subheader("The AI's Prediction:")
    st.title(predicted_class)

    st.subheader("Confidence Score:")
    # Display the confidence for each possible digit as a bar chart
    confidence_dict = dict(zip(range(10), prediction[0]))
    st.bar_chart(confidence_dict)
