import streamlit as st
import tensorflow as tf
import numpy as np


# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("saved_model_98.15_256.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. *Upload Image:* Go to the *Disease Recognition* page and upload an image of a plant with suspected diseases.
    2. *Analysis:* Our system will process the image using advanced algorithms to identify potential diseases.
    3. *Results:* View the results and recommendations for further action.

    ### Why Choose Us?gi
    - *Accuracy:* Our system utilizes state-of-the-art Deep Learning techniques for accurate disease detection.
    - *User-Friendly:* Simple and intuitive interface for seamless user experience.
    - *Fast and Efficient:* Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the *Disease Recognition* page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the *About* page.
    """)

# About Project
elif app_mode == "About":
    st.header("About Us")
    st.markdown("""
               
               

### Welcome to our project website, dedicated to advancing agricultural technology through innovative solutions!

We Charanya A, Gowthami GS, Mahesh R and Yogananda KS, enthusiastic students from SIDDAGANGA INSTITUTE OF TECHNOLOGY, with a deep passion for data science. Driven by our keen interest in harnessing the power of technology to solve real-world problems, we have embarked on a mission to improve the agricultural sector.

Our project focuses on *early potato disease detection* using cutting-edge deep learning techniques. By integrating IoT sensors for precise soil analysis, we aim to revolutionize farming practices and enhance crop yield and quality. Our approach ensures that diseases are identified at the earliest stages, allowing for timely intervention and minimizing crop loss.

This project has been a journey of discovery and innovation, guided by the invaluable mentorship of Dr.Jagadamaba G. We extend our heartfelt thanks to her for her unwavering support and guidance, which have been instrumental in the successful realization of our vision.

Our website is designed with the hope of empowering farmers by providing them with advanced tools and insights for better crop management. Through our platform, we aspire to contribute to the agricultural community by offering practical solutions that lead to more efficient and sustainable farming practices.

Thank you for visiting our website. We are excited to share our work with you and hope it makes a meaningful impact in the field of agriculture. Together, let's cultivate a future where technology and farming go hand in hand for a better tomorrow.
                """)

           
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")

    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)

        if st.button("Predict"):
            result_index = model_prediction(test_image)
            class_names = ['Early_Blight', 'Healthy', 'Late_Blight', 'Non_Potato', 'Object']
            st.success(f"Model successfully predicts it's {class_names[result_index]}")
            
        