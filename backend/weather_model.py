
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os

# Weather types we want to predict
WEATHER_TYPES = ["Sunny", "Partly Cloudy", "Cloudy", "Rainy", "Thunderstorm", "Snowy", "Drizzle", "Foggy"]

def create_synthetic_weather_data(num_samples=1000):
    """Generate synthetic weather data for training"""
    np.random.seed(42)
    
    # Generate features with realistic ranges and correlations
    temperature = np.random.uniform(-15, 45, num_samples)  # in Celsius
    humidity = np.random.uniform(20, 100, num_samples)  # in %
    pressure = np.random.normal(1013, 10, num_samples)  # in hPa
    wind_speed = np.random.exponential(10, num_samples)  # in km/h
    wind_speed = np.clip(wind_speed, 0, 150)
    precipitation = np.zeros(num_samples)  # in mm
    
    # Add correlations and create weather types
    weather_types = []
    for i in range(num_samples):
        if temperature[i] > 30 and humidity[i] < 50:
            weather_types.append("Sunny")
            # Sunny days have low precipitation
            precipitation[i] = np.random.exponential(0.5)
        elif temperature[i] < 0 and humidity[i] > 70:
            weather_types.append("Snowy")
            # Snowy days have moderate precipitation
            precipitation[i] = np.random.uniform(1, 10)
        elif humidity[i] > 85 and wind_speed[i] < 10:
            weather_types.append("Foggy")
            precipitation[i] = np.random.exponential(1)
        elif humidity[i] > 80 and temperature[i] > 20 and wind_speed[i] > 30:
            weather_types.append("Thunderstorm")
            # Thunderstorms have high precipitation
            precipitation[i] = np.random.uniform(10, 50)
        elif humidity[i] > 70 and temperature[i] > 15:
            # Rainy or drizzle based on precipitation amount
            if np.random.random() > 0.7:
                weather_types.append("Rainy")
                precipitation[i] = np.random.uniform(5, 30)
            else:
                weather_types.append("Drizzle")
                precipitation[i] = np.random.uniform(1, 5)
        elif humidity[i] > 60 and temperature[i] > 18:
            weather_types.append("Partly Cloudy")
            precipitation[i] = np.random.exponential(0.8)
        else:
            weather_types.append("Cloudy")
            precipitation[i] = np.random.exponential(2)
    
    # Create DataFrame
    df = pd.DataFrame({
        'temperature': temperature,
        'humidity': humidity,
        'pressure': pressure,
        'wind_speed': wind_speed,
        'precipitation': precipitation,
        'weather_type': weather_types
    })
    
    return df

def train_weather_model():
    """Train a RandomForest model on the synthetic data"""
    # Generate or load data
    df = create_synthetic_weather_data(2000)
    
    # Split features and target
    X = df[['temperature', 'humidity', 'pressure', 'wind_speed', 'precipitation']]
    y = df['weather_type']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a pipeline with preprocessing and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    accuracy = pipeline.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")
    
    # Save the model
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(pipeline, 'models/weather_model.joblib')
    
    return pipeline, accuracy

def load_model():
    """Load the trained model or train if it doesn't exist"""
    model_path = 'models/weather_model.joblib'
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            return model
        else:
            model, _ = train_weather_model()
            return model
    except Exception as e:
        print(f"Error loading model: {e}")
        model, _ = train_weather_model()
        return model

def predict_weather(temperature, humidity, pressure, wind_speed, precipitation):
    """Make a prediction using the trained model"""
    model = load_model()
    
    # Create a feature array
    features = np.array([[temperature, humidity, pressure, wind_speed, precipitation]])
    
    # Get the predicted weather type
    weather_type = model.predict(features)[0]
    
    # Get probability scores
    probabilities = model.predict_proba(features)[0]
    probability = max(probabilities)
    
    return weather_type, probability

if __name__ == "__main__":
    # Train the model when script is run directly
    train_weather_model()
