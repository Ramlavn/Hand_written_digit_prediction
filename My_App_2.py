import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load your trained model
model = tf.keras.models.load_model('my_model.h5')

# Function to preprocess the image to the model's input shape
def preprocess_image(image):
    # Convert to grayscale
    image = image.convert('L')
    
    # Invert colors if the background is black
    if np.mean(np.array(image)) < 128:
        image = Image.fromarray(255 - np.array(image))
    
    # Resize with LANCZOS resampling
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to array and normalize
    image = np.array(image).astype('float32') / 255.0
    
    # Ensure black background and white digit
    image = 1 - image
    
    # Reshape for model input
    image = image.reshape(1, 28, 28, 1)
    
    return image

# Streamlit App
st.title("Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit or use the drawing tool, and the model will predict the digit.")

# Include your name, LinkedIn, and GitHub profiles
st.write("### Author: Ramlavan")
st.write("Connect with me:")
st.markdown("[LinkedIn Profile](https://www.linkedin.com/in/ramlavan-arumugasamyi-13b046296/)")
st.markdown("[GitHub Profile](https://github.com/Ramlavn)")

# Option to upload file or draw
option = st.radio("Choose input method:", ("Upload Image", "Draw Digit"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
elif option == "Draw Digit":
    canvas_result = st_canvas(
        stroke_width=10,
        stroke_color="#ffffff",
        background_color="#000000",
        height=280,
        width=280,
        key="canvas"
    )
    if canvas_result.image_data is not None:
        image = Image.fromarray(canvas_result.image_data.astype('uint8'))
        st.image(image, caption='Drawn Image.', use_column_width=True)

if 'image' in locals():
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Predict using the model
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)
    confidence_score = np.max(prediction)
    
    # Display the predicted digit and confidence score
    st.write(f"Predicted Digit: {predicted_digit}")
    st.write(f"Confidence Score: {confidence_score:.2f}")
    
    # Show confidence as a percentage
    confidence_percentage = confidence_score * 100
    st.write(f"Confidence Percentage: {confidence_percentage:.2f}%")
    
    # Add a note
    st.write("Note: Results may vary based on image quality and clarity. For best results, draw clear, centered digits.")
