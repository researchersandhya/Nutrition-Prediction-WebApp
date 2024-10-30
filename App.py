import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

# Function to download the model from Google Drive
def download_model():
    file_id = '1uLPt0uilJmyuDNfEgcKUi7H8jDO1bFGp'  # Replace with your actual file ID
    url = f'https://drive.google.com/uc?id={file_id}'
    response = requests.get(url)
    with open("trained_nutrition_model.keras", "wb") as file:
        file.write(response.content)

# Load nutrition data
df = pd.read_csv('nutrition.csv')

# Download the model
download_model()

# Load TensorFlow model
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_nutrition_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Inject custom CSS for better visuals
st.markdown(""" 
    <style> 
    body { 
        background-color: #f3f4f6; 
    } 
    .header { 
        color: #ff6347; 
        font-family: 'Helvetica Neue', sans-serif; 
        text-align: center; 
        font-size: 3em; 
        font-weight: bold; 
        padding: 0.5em; 
    } 
    .main-content { 
        font-family: 'Arial', sans-serif; 
        margin: 20px; 
        padding: 20px; 
    } 
    .nutrition-facts { 
        font-size: 1.2em; 
        color: #2e2e2e; 
        line-height: 1.6; 
    } 
    .nutrition-facts strong { 
        font-weight: bold; 
    } 
    </style> 
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Nutrition Prediction"])

# Home Page
if app_mode == "Home":
    st.markdown('<div class="header">NUTRITION PREDICTION SYSTEM</div>', unsafe_allow_html=True)
    st.image("nutrition_dashboard.jpg", use_column_width=True)
    st.markdown("""
        Welcome to the **Nutrition Value Prediction System!** 🍽️🔍
        
        Upload a food image, and the system will predict the nutrition content of your meal.

        ### Steps to use:
        1. Upload a food image.
        2. The system predicts nutrition values.
        3. Get a detailed report with calories, proteins, fats, and carbs!

        **Let’s make healthy food choices easier.**
    """)

# About Page
elif app_mode == "About":
    st.header("About the Project")
    st.markdown("""
        ### Dataset Information
        This project uses the Food-101 dataset which consists of 87,000 food images across 101 different categories.
        The goal is to predict the nutritional value of food items from images using machine learning.
    """)

# Prediction Page
elif app_mode == "Nutrition Prediction":
    st.header("Nutrition Prediction")
    test_image = st.file_uploader("Upload an Image of the food:", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        # Show uploaded image
        st.image(test_image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            result_index = model_prediction(test_image)
            
            # Food classes
            class_names = [
                'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beet_salad', 'bread_pudding',
                'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cheesecake', 'chicken_curry', 'chicken_wings', 
                'chocolate_cake', 'clam_chowder', 'club_sandwich', 'creme_brulee', 'croque_madame', 'cup_cakes', 
                'donuts', 'edamame', 'falafel', 'french_fries', 'french_toast', 'gnocchi', 'greek_salad', 
                'hamburger', 'hot_dog', 'ice_cream', 'lasagna', 'lobster_roll_sandwich', 'macarons', 'nachos', 
                'omelette', 'pancakes', 'pizza', 'ramen', 'red_velvet_cake', 'risotto', 'sashimi', 'spaghetti_bolognese', 
                'steak', 'sushi', 'tiramisu', 'waffles'
            ]
            
            predicted_food = class_names[result_index]
            st.success(f"The model predicts this is: {predicted_food}")

            # Get nutrition info
            class_name = class_names[result_index]
            filtered_data = df[df['Food'] == class_name]

            if not filtered_data.empty:
                nutrition_info = filtered_data.iloc[0]

                # Display nutrition values with better layout
                st.subheader(f"Nutrition Information for {class_name}")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**Calories:** {nutrition_info['Calories (kcal)']} kcal")
                    st.markdown(f"**Protein:** {nutrition_info['Protein (g)']} g")
                    st.markdown(f"**Fat:** {nutrition_info['Fat (g)']} g")

                with col2:
                    st.markdown(f"**Carbohydrates:** {nutrition_info['Carbs (g)']} g")

                # Pie chart for visual representation
                fig, ax = plt.subplots()
                nutrition_values = [nutrition_info['Protein (g)'], nutrition_info['Fat (g)'], nutrition_info['Carbs (g)']]
                labels = ['Protein', 'Fat', 'Carbohydrates']
                ax.pie(nutrition_values, labels=labels, autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'], startangle=90)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig)
            else:
                st.warning("No nutritional data available for this food item.")
