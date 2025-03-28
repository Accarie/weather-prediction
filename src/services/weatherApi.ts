
import { WeatherInputs } from "@/components/WeatherInputForm";
import { PredictionResult } from "@/components/WeatherPredictionResult";

// Mock weather types for demonstration
const weatherTypes = [
  "Sunny", 
  "Partly Cloudy", 
  "Cloudy", 
  "Rainy", 
  "Thunderstorm", 
  "Snowy",
  "Drizzle",
  "Foggy"
];

// Days of the week
const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const today = new Date().getDay();
const futureDays = [...days.slice(today), ...days.slice(0, today)].slice(1, 6);

// In a real implementation, this would connect to your Python ML model API
export const predictWeather = async (inputs: WeatherInputs): Promise<PredictionResult> => {
  // Simulate API call delay
  return new Promise((resolve) => {
    setTimeout(() => {
      // Mock logic to determine weather type based on inputs
      // In a real implementation, this would call your Python model API
      
      let predictedType = "Cloudy"; // default
      let tempModifier = 0;
      
      if (inputs.temperature > 30 && inputs.humidity < 50) {
        predictedType = "Sunny";
        tempModifier = 2;
      } else if (inputs.precipitation > 5) {
        predictedType = "Rainy";
        tempModifier = -2;
      } else if (inputs.precipitation > 1 && inputs.precipitation <= 5) {
        predictedType = "Drizzle";
        tempModifier = -1;
      } else if (inputs.humidity > 80 && inputs.temperature < 5) {
        predictedType = "Snowy";
        tempModifier = -4;
      } else if (inputs.humidity > 90 && inputs.windSpeed < 5) {
        predictedType = "Foggy";
        tempModifier = -1;
      } else if (inputs.precipitation > 10 && inputs.windSpeed > 30) {
        predictedType = "Thunderstorm";
        tempModifier = -3;
      } else if (inputs.humidity > 60 && inputs.temperature > 20) {
        predictedType = "Partly Cloudy";
        tempModifier = 0;
      }
      
      // Generate mock forecast data
      const forecast = futureDays.map((day, index) => {
        // Create some variation in the forecast
        const variance = Math.sin(index) * 3;
        const tempHigh = Math.round(inputs.temperature + tempModifier + variance + index * 0.5);
        const tempLow = Math.round(tempHigh - (3 + Math.random() * 5));
        
        // Sometimes change the weather type for the forecast
        const weatherIndex = (weatherTypes.indexOf(predictedType) + index) % weatherTypes.length;
        const weatherType = Math.random() > 0.7 ? weatherTypes[weatherIndex] : predictedType;
        
        return {
          day,
          weatherType,
          tempHigh,
          tempLow
        };
      });
      
      // Generate a confidence score between 0.7 and 0.99
      const confidence = 0.7 + Math.random() * 0.29;
      
      resolve({
        weatherType: predictedType,
        probability: confidence,
        temperature: Math.round(inputs.temperature + tempModifier),
        conditions: {
          humidity: inputs.humidity,
          precipitation: inputs.precipitation,
          windSpeed: inputs.windSpeed
        },
        forecast
      });
    }, 1500); // Simulate 1.5s delay
  });
};

// Function to prepare chart data from prediction results
export const prepareChartData = (result: PredictionResult) => {
  if (!result) return null;
  
  return result.forecast.map((day, index) => ({
    day: day.day,
    temperature: (day.tempHigh + day.tempLow) / 2,
    humidity: result.conditions.humidity + (Math.sin(index) * 10),
    precipitation: result.conditions.precipitation * (Math.cos(index) * 0.5 + 0.5)
  }));
};
