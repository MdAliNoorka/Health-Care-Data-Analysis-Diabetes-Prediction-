import streamlit as st
import pandas as pd
import joblib
import os
import tensorflow as tf

# Set app title
st.title('Diabetes Prediction App')

# Scan models directory for available models
@st.cache_resource
def get_available_models():
    models_dir = 'models'
    models_dict = {}
    
    if not os.path.exists(models_dir):
        st.warning(f"Models directory '{models_dir}' not found.")
        return {}
    
    for file in os.listdir(models_dir):
        file_path = os.path.join(models_dir, file)
        if file.endswith('.pkl'):
            model_name = file.replace('.pkl', '').replace('_', ' ').title()
            models_dict[model_name] = file_path
        elif file.endswith('.h5') or file.endswith('.keras'):
            model_name = file.split('.')[0].replace('_', ' ').title()
            models_dict[model_name] = file_path
    
    return models_dict

# Get available models
available_models = get_available_models()

if not available_models:
    st.error("No models found in the models directory. Please add models to continue.")
    st.stop()

# Model selection dropdown in sidebar
st.sidebar.header('Model Selection')
selected_model_name = st.sidebar.selectbox(
    'Choose a prediction model:',
    list(available_models.keys())
)
selected_model_path = available_models[selected_model_name]

# Load the selected model
@st.cache_resource
def load_model(model_path):
    try:
        if model_path.endswith('.pkl'):
            return joblib.load(model_path)
        elif model_path.endswith('.h5') or model_path.endswith('.keras'):
            return tf.keras.models.load_model(model_path)
        else:
            st.error(f"Unsupported model format: {model_path}")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model(selected_model_path)
if model is None:
    st.error("Failed to load the selected model.")
    st.stop()
else:
    st.sidebar.success(f"Successfully loaded: {selected_model_name}")

# Add input widgets
st.header('Patient Information')
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0)
    glucose = st.number_input('Glucose (mg/dL)', min_value=0, max_value=200, value=100)
    blood_pressure = st.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=122, value=70)
    skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20)
    
with col2:
    insulin = st.number_input('Insulin (μU/mL)', min_value=0, max_value=900, value=80)
    bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input('Age', min_value=0, max_value=120, value=25)

# Create feature dataframe
features = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [blood_pressure],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [dpf],
    'Age': [age]
})

# Make prediction
if st.button('Predict'):
    try:
        # Handle different model types
        if selected_model_path.endswith('.h5') or selected_model_path.endswith('.keras'):
            # For neural network models
            raw_prediction = model.predict(features)[0][0]
            probability = float(raw_prediction)
            prediction = 1 if probability >= 0.5 else 0
        else:
            # For sklearn models
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1]
        
        st.subheader('Results')
        if prediction == 1:
            st.error(f'High risk of diabetes (Probability: {probability:.2%})')
        else:
            st.success(f'Low risk of diabetes (Probability: {probability:.2%})')
        
        # Display model information
        st.sidebar.info(f'Prediction made using {selected_model_name}')
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.sidebar.error("The selected model may not be compatible with the input format.")