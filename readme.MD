# Diabetes Prediction System

## Overview
This project is a diabetes prediction system that uses machine learning models to predict the likelihood of diabetes based on several medical indicators. The system includes data preprocessing pipelines, exploratory data analysis (EDA), model training, and a Streamlit web application for easy prediction.

## Project Structure
```
├── app.py                    # Streamlit web application
├── code.ipynb                # Main Jupyter notebook with data preprocessing and EDA
├── data.csv                  # Raw dataset
├── processed_dataset.csv     # Preprocessed dataset
├── req.txt                   # Requirements file
└── models/                   # Directory containing trained models
    ├── logistic_regression_model.pkl
    ├── neural_network_model.keras
    └── random_forest_model.pkl
```

## Features
The system predicts diabetes based on the following features:
- Pregnancies
- Glucose level
- Blood Pressure
- Skin Thickness
- Insulin level
- BMI (Body Mass Index)
- Diabetes Pedigree Function
- Age

## Data Processing Pipeline
The project follows a structured data processing pipeline:
1. **Data Loading**: Load the dataset from CSV files
2. **Data Exploration**: Initial examination of data characteristics
3. **Data Cleaning**: Handle missing values and outliers
4. **Data Transformation**: Scale numerical features and encode categorical variables

## Exploratory Data Analysis (EDA)
The EDA process includes:
- Univariate analysis (distribution of individual features)
- Bivariate and multivariate analysis (relationships between features)
- Feature importance analysis using mutual information scores

## Machine Learning Models
The system includes multiple machine learning models:
- Logistic Regression
- Random Forest
- Neural Network

## Web Application
The Streamlit-based web application provides:
- Model selection from available trained models
- Input form for patient information
- Prediction display with probability score
- Error handling for incompatible models or incorrect inputs

## Installation and Setup
1. Clone the repository
2. Install required packages:
   ```
   pip install -r req.txt
   ```
3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Usage
1. Select a prediction model from the sidebar
2. Enter patient information in the input fields
3. Click "Predict" to get the diabetes risk assessment

## Requirements
- streamlit
- pandas
- scikit-learn
- joblib
- tensorflow (for neural network models)

## Project Background
This project was developed as part of a Data Science course at FAST National University to demonstrate the application of machine learning techniques in healthcare diagnostics.