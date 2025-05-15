import pickle
import numpy as np
import streamlit as st

# Load the trained ML model
try:
    asd_model = pickle.load(open(r"C:\Users\likit\Downloads\ASD final\ASD final\ASD\ASD.sav", 'rb'))
except Exception as e:
    st.error("Error loading the model. Ensure the file exists and is not corrupted.")
    asd_model = None

# Relation Encoding Dictionary
relation_mapping = {"Self": 0, "Parent": 1, "Relative": 2}

# Age Description Encoding Dictionary with clear definitions
age_desc_mapping = {
    "Infant (0-12 months)": 0,
    "Toddler (1-3 years)": 1,
    "Child (4-12 years)": 2,
    "Teen (13-19 years)": 3,
    "Adult (20+ years)": 4
}

# Ethnicity Encoding Dictionary
ethnicity_mapping = {
    "White-European": 0,
    "Black": 1,
    "Asian": 2,
    "Middle Eastern": 3,
    "South Asian": 4,
    "Hispanic": 5,
    "Latino": 6,
    "Other": 7
}

# Streamlit UI with multiple pages
st.set_page_config(page_title="ASD Prediction App", page_icon="🤖", layout="wide")

# Sidebar navigation
menu = st.sidebar.selectbox("Menu", ["Home", "Predict ASD", "About ASD"])

if menu == "Home":
    st.title("Welcome to the ASD Prediction App")
    st.image(r"C:\Users\likit\Downloads\ASD final\ASD final\ASD\images.jpg", caption="Autism Spectrum Disorder Awareness")
    st.write("""
    Autism Spectrum Disorder (ASD) is a developmental disorder that affects communication and behavior.
    It is known as a spectrum disorder because individuals can experience a wide range of symptoms and severity levels.
    
    This application allows users to predict the likelihood of ASD based on input symptoms and personal details.
    
    Click on the 'Predict ASD' page in the menu to start the prediction process.
    """)

elif menu == "About ASD":
    st.title("About Autism Spectrum Disorder (ASD)")
    st.write("""
    **What is Autism Spectrum Disorder (ASD)?**
    - ASD is a neurological and developmental disorder that begins in early childhood and lasts throughout a person's life.
    - Common symptoms include difficulty with communication, repetitive behaviors, and restricted interests.
    - Early diagnosis and intervention can help individuals with ASD improve their quality of life.
    
    **Causes and Risk Factors:**
    - Genetic factors
    - Environmental influences
    - Differences in brain structure and function
    
    **How is ASD Diagnosed?**
    - Clinical assessments and behavioral observations
    - Standardized ASD screening tests
    """)

elif menu == "Predict ASD":
    st.title("ASD Prediction Using Machine Learning")
    st.write("""This application helps predict the likelihood of Autism Spectrum Disorder (ASD) based on user inputs.""")

    # User input fields
    symptoms_list = [
        "Does the individual avoid eye contact?", 
        "Does the individual engage in repetitive movements (e.g., hand flapping, rocking)?", 
        "Does the individual have difficulty understanding social cues?",
        "Does the individual show an intense interest in specific topics?", 
        "Does the individual have speech or language development issues?",
        "Is the individual overly sensitive to sensory input (e.g., sounds, textures, lights)?", 
        "Does the individual have a strong preference for routine and resist change?", 
        "Does the individual tend to withdraw from social situations or prefer isolation?",
        "Does the individual show unusual emotional reactions?", 
        "Does the individual have an interest in making friends?"
    ]

    user_inputs = [st.radio(f"{symptom}", ["No", "Yes"], index=0, horizontal=True) for symptom in symptoms_list]
    user_inputs = [1 if val == "Yes" else 0 for val in user_inputs]

    age = st.number_input("What is the individual's age?", min_value=1, max_value=100, step=1)
    gender = st.radio("What is the individual's gender?", ["Male", "Female"], horizontal=True)
    gender = 1 if gender == "Male" else 0
    jaundice = st.radio("Did the individual have neonatal jaundice?", ["No", "Yes"], horizontal=True)
    jaundice = 1 if jaundice == "Yes" else 0
    used_app_before = st.radio("Has the individual used a diagnostic app before?", ["No", "Yes"], horizontal=True)
    used_app_before = 1 if used_app_before == "Yes" else 0
    autism = st.radio("Is there a family history of ASD?", ["No", "Yes"], horizontal=True)
    autism = 1 if autism == "Yes" else 0

    ethnicity = st.radio("What is the individual's ethnicity?", list(ethnicity_mapping.keys()), horizontal=True)
    ethnicity = ethnicity_mapping[ethnicity]

    relation = st.radio("What is your relation to the individual?", list(relation_mapping.keys()), horizontal=True)
    relation = relation_mapping[relation]

    age_desc = st.radio("Which age category best describes the individual?", list(age_desc_mapping.keys()), horizontal=True)
    age_desc = age_desc_mapping[age_desc]

    contry_of_res = 0  # Placeholder since it was dropped in training
    result = 0  # Placeholder to ensure 20 features


    #prediction
    if st.button("Get ASD Prediction"):
        if asd_model is not None:
            try:
                input_data = np.array(user_inputs + [
                    float(age), float(gender), float(ethnicity), float(jaundice), float(autism),
                    float(used_app_before), float(age_desc), float(relation), float(contry_of_res), float(result)
                ]).reshape(1, -1)
                
                prediction_proba = asd_model.predict_proba(input_data)[0]
                confidence = prediction_proba[1] * 100
                prediction = 1 if prediction_proba[1] > 0.3 else 0
                result_text = "Positive for ASD" if prediction == 1 else "Negative for ASD"
                st.success(f'The model predicts: {result_text} with {confidence:.2f}% confidence')
                
                if prediction == 1:
                    st.write("### Recommended Treatments:")
                    st.write("- **Behavioral Therapy**: Helps improve communication, social skills, and behavior.")
                    st.write("- **Speech Therapy**: Supports language development and communication skills.")
                    st.write("- **Occupational Therapy**: Assists with daily living activities and sensory integration.")
                    st.write("- **Early Intervention Programs**: Structured programs to enhance learning and development.")
            except Exception as e:
                st.error(f"Error in prediction: {e}")
        else:
            st.error("Model is not loaded. Please check the file path and try again.")
