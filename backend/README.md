
# Weather Prediction ML Service

This is a machine learning service that predicts weather conditions based on input parameters like temperature, humidity, pressure, wind speed, and precipitation.

## Features

- Machine Learning model built with scikit-learn
- Uses real-world weather data from NOAA GSOD dataset
- FastAPI backend for predictions
- 5-day weather forecast generation

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Download and process the dataset:
   ```
   python dataset_loader.py
   ```

3. Train the model:
   ```
   python weather_model.py
   ```

4. Start the API server:
   ```
   uvicorn main:app --reload
   ```

The API will be available at `http://localhost:8000`.

### Docker

You can also run the service using Docker:

```
docker build -t weather-prediction .
docker run -p 8000:8000 weather-prediction
```

## Dataset

The model uses the NOAA Global Surface Summary of the Day (GSOD) dataset, which contains historical weather data from weather stations around the world. The data is processed to extract relevant features like temperature, humidity, pressure, wind speed, and precipitation.

## API Endpoints

- `GET /`: Check if the API is running
- `POST /predict`: Get a weather prediction. Requires a JSON body with temperature, humidity, pressure, wind_speed, and precipitation.
- `GET /train`: Retrain the model with updated data

## Example Usage

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "temperature": 25,
    "humidity": 60,
    "pressure": 1013,
    "wind_speed": 10,
    "precipitation": 0
}

response = requests.post(url, json=data)
prediction = response.json()
print(prediction)
```

