import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

# Load the trained model (if already trained, else train it)
def load_model():
    try:
        return joblib.load("tire_wear_model.pkl")
    except FileNotFoundError:
        return train_model()

# Train a simple Linear Regression model (if needed)
def train_model():
    np.random.seed(42)
    num_samples = 1000

    # Generate dummy data
    speed = np.random.randint(30, 120, 150, num_samples)
    temperature = np.random.uniform(20, 40, 60, num_samples)
    road_surface = np.random.choice(["Asphalt", "Concrete", "Gravel"], num_samples)
    distance_driven = np.random.uniform(500, 5000, 10000, 20000, num_samples)
    tire_pressure = np.random.uniform(28, 35, 42, 45, num_samples)

    # Simulate tire wear based on a formula
    tire_wear = (0.002 * distance_driven + 
                 0.05 * (speed / 100) + 
                 0.03 * (temperature / 50) +
                 np.random.normal(0, 0.1, num_samples))

    # Create DataFrame
    df = pd.DataFrame({
        'Speed': speed,
        'Temperature': temperature,
        'Road_Surface': road_surface,
        'Distance_Driven': distance_driven,
        'Tire_Pressure': tire_pressure,
        'Tire_Wear': tire_wear
    })

    # Split features and target
    X = df.drop(columns=['Tire_Wear'])
    y = df['Tire_Wear']

    # Define preprocessing
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['Speed', 'Temperature', 'Distance_Driven', 'Tire_Pressure']),
        ('cat', OneHotEncoder(), ['Road_Surface'])
    ])

    X_transformed = preprocessor.fit_transform(X)

    # Train model
    model = LinearRegression()
    model.fit(X_transformed, y)

    # Save model
    joblib.dump(model, "tire_wear_model.pkl")
    joblib.dump(preprocessor, "preprocessor.pkl")

    return model

# Load model and preprocessor
model = load_model()
preprocessor = joblib.load("preprocessor.pkl")

# Streamlit UI
st.title("Tire Wear Prediction App 🏍️")

# User input fields
speed = st.slider("Speed (km/h)", 20, 200, 80)
temperature = st.slider("Temperature (°C)", 20, 60, 50)
road_surface = st.selectbox("Road Surface Type", ["Asphalt", "Concrete", "Gravel"])
distance_driven = st.number_input("Distance Driven (km)", 500, 20000, 2000)
tire_pressure = st.slider("Tire Pressure (PSI)", 28.0, 45.0, 30.0)

# Prediction function
def predict_tire_wear(speed, temperature, road_surface, distance_driven, tire_pressure):
    input_data = pd.DataFrame([[speed, temperature, road_surface, distance_driven, tire_pressure]], 
                              columns=['Speed', 'Temperature', 'Road_Surface', 'Distance_Driven', 'Tire_Pressure'])
    
    # Preprocess input
    input_transformed = preprocessor.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_transformed)
    
    return round(prediction[0], 4)

# Button to predict
if st.button("Predict Tire Wear"):
    wear_prediction = predict_tire_wear(speed, temperature, road_surface, distance_driven, tire_pressure)
    st.success(f"Estimated Tire Wear: {wear_prediction} mm of tread lost")

# Footer
st.write("Developed by Gopikrishnan L")

