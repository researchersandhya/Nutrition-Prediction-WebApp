import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
df = pd.read_csv('nutrition.csv')
#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_nutrition_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Nutrition Prediction"])

#Main Page
if(app_mode=="Home"):
    st.header("NUTRITION PREDICTION SYSTEM")
    #image_path = "home_page.jpeg"
    #st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Nutrition Value Prediction System! 🍽️🔍

    Our mission is to help you understand the nutritional content of your food efficiently. Upload an image of a dish, and our system will analyze it to predict its nutrition values. Together, let's make informed dietary choices for a healthier lifestyle!

    ### How It Works
    1. **Upload Image:** Go to the **Nutrition Prediction** page and upload an image of the food item.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify the food and calculate its nutritional values.
    3. **Results:** View the predicted nutrition information including calories, proteins, fats, and carbohydrates.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for precise nutrition prediction.
    - **User-Friendly:** Simple and intuitive interface for a seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, enabling quick dietary assessments.

    ### Get Started
    Click on the **Nutrition Prediction** page in the sidebar to upload an image and experience the power of our Nutrition Value Prediction System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
""")


#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
This dataset is created using offline augmentation from the original Food-101 dataset.

This dataset consists of approximately 87,000 RGB images of various food items categorized into 101 different classes. The total dataset is divided into an 80/20 ratio for training and validation, while preserving the directory structure. Additionally, a new directory containing 33 test images has been created for prediction purposes.

#### Content
1. **Train** (70,295 images)
2. **Validation** (17,572 images)
3. **Test** (33 images)


                """)

#Prediction Page
elif(app_mode=="Nutrition Prediction"):
    st.header("Nutrition Prediction")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        #st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_names = [
    'apple_pie',
    'baby_back_ribs',
    'baklava',
    'beef_carpaccio',
    'beef_tartare',
    'beet_salad',
    'beignets',
    'bibimbap',
    'bread_pudding',
    'breakfast_burrito',
    'bruschetta',
    'caesar_salad',
    'cannoli',
    'caprese_salad',
    'carrot_cake',
    'ceviche',
    'cheese_plate',
    'cheesecake',
    'chicken_curry',
    'chicken_quesadilla',
    'chicken_wings',
    'chocolate_cake',
    'chocolate_mousse',
    'churros',
    'clam_chowder',
    'club_sandwich',
    'crab_cakes',
    'creme_brulee',
    'croque_madame',
    'cup_cakes',
    'deviled_eggs',
    'donuts',
    'dumplings',
    'edamame',
    'eggs_benedict',
    'escargots',
    'falafel',
    'filet_mignon',
    'fish_and_chips',
    'foie_gras',
    'french_fries',
    'french_onion_soup',
    'french_toast',
    'fried_calamari',
    'fried_rice',
    'frozen_yogurt',
    'garlic_bread',
    'gnocchi',
    'greek_salad',
    'grilled_cheese_sandwich',
    'grilled_salmon',
    'guacamole',
    'gyoza',
    'hamburger',
    'hot_and_sour_soup',
    'hot_dog',
    'huevos_rancheros',
    'hummus',
    'ice_cream',
    'lasagna',
    'lobster_bisque',
    'lobster_roll_sandwich',
    'macaroni_and_cheese',
    'macarons',
    'miso_soup',
    'mussels',
    'nachos',
    'omelette',
    'onion_rings',
    'oysters',
    'pad_thai',
    'paella',
    'pancakes',
    'panna_cotta',
    'peking_duck',
    'pho',
    'pizza',
    'pork_chop',
    'poutine',
    'prime_rib',
    'pulled_pork_sandwich',
    'ramen',
    'ravioli',
    'red_velvet_cake',
    'risotto',
    'samosa',
    'sashimi',
    'scallops',
    'seaweed_salad',
    'shrimp_and_grits',
    'spaghetti_bolognese',
    'spaghetti_carbonara',
    'spring_rolls',
    'steak',
    'strawberry_shortcake',
    'sushi',
    'tacos',
    'takoyaki',
    'tiramisu',
    'tuna_tartare',
    'waffles'
]
       


        st.success("Model is Predicting it's a {}".format(class_names[result_index]))
        # st.success("It's nutrition values are {}".format(df[df['Food'] == class_names[result_index]]))

        # Define the class name you want to filter by
        class_name = class_names[result_index]

        # Filter the DataFrame for the specific class   name
        filtered_data = df[df['Food'] == class_name]

        if not filtered_data.empty:
            # Extract the nutrition values
            nutrition_info = filtered_data.iloc[0]  # Get the first matching row

            # Create a formatted string with nutrition values
            nutrition_values = (
                f"**Calories:** {nutrition_info['Calories (kcal)']} kcal\n"
                f"\n**Protein:** {nutrition_info['Protein (g)']} g\n"
                f"\n**Fat:** {nutrition_info['Fat (g)']} g\n"
                f"\n**Carbohydrates:** {nutrition_info['Carbs (g)']} g"
            )

            st.success(f"It's nutrition values are:\n\n{nutrition_values}")
        else:
            st.warning("No data found for this food item.")
