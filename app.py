import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from PIL import Image

# Define label dictionary
labels = {
    0: "Speed Limit 20", 1: "Speed Limit 30", 2: "Speed Limit 50", 3: "Speed Limit 60", 4: "Speed Limit 70",
    5: "Speed Limit 80", 6: "End of a Speed Limit 80", 7: "Speed Limit 100", 8: "Speed Limit 120", 9: "Speed Limit 100",
    10: "No overtaking by trucks", 11: "Crossroads", 12: "Priority Road", 13: "Give way", 14: "Stop",
    15: "All vehicles prohibited in both directions", 16: "No trucks", 17: "No Entry", 18: "Other Hazards", 19: "Curve to left",
    20: "Curve to right", 21: "Double curve, first to the left", 22: "Uneven Road", 23: "Slippery Road", 24: "Road Narrows Near Side",
    25: "Roadworks", 26: "Traffic lights", 27: "No pedestrians", 28: "Children", 29: "Cycle Route",
    30: "Be careful in winter", 31: "Wild animals", 32: "No parking", 33: "Turn right ahead", 34: "Turn left ahead",
    35: "Ahead Only", 36: "Proceed straight or turn right", 37: "Proceed straight or turn left", 38: "Pass onto right",
    39: "Pass onto left", 40: "Roundabout", 41: "No overtaking", 42: "End of Truck Overtaking Prohibition"
}

# Load the trained model
model = load_model("model.keras")

# Streamlit UI for image upload and prediction
st.title("Traffic Sign Recognition 🚦")

# Introduction Section
st.markdown("""
### Introduction
This model recognizes traffic signs using a deep learning approach. 
The dataset used for training the model is the **German Traffic Sign Recognition Benchmark (GTSRB)**, which consists of 43 different traffic sign classes. 
This model helps identify traffic signs and their corresponding meanings, which can be useful in building autonomous driving systems and other safety applications.

#### How to use:
- Upload an image of a Traffic Sign.
- The model will predict Traffic Sign.
""")

# Upload image using Streamlit (both upload button & drag-and-drop)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
st.write("Or drag and drop an image into the box above.")

if uploaded_file is not None:
    # Open the image
    try:
        img = Image.open(uploaded_file)
        img = img.convert('RGB')  # Ensure the image is in RGB mode
        img = img.resize((30, 30))  # Resize image to 30x30 as expected by the model
    except Exception as e:
        st.error(f"Error loading image: {e}")
    
    # Preprocess the image
    img_array = np.array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize and add batch dimension
    
    # Predict the class
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]  # Get the class with the highest probability
    predicted_label = labels[predicted_class]  # Map class to label
    
    # Output Section: Display image and prediction
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write(f"     **Prediction** - {predicted_label}")
    
# Conclusion Section
st.markdown("""
### Conclusion
This model has been trained on a diverse set of traffic signs, achieving high accuracy in predicting the correct sign. The accuracy of the model depends on factors such as the quality of the image and the clarity of the sign in the image. This model can help in applications such as autonomous driving systems, road safety applications, and assistive technologies for drivers.
""")

# streamlit run "F:/DL_Project/app.py"