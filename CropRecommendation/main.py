import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Load dataset
df = pd.read_csv("/Users/ishaan/Documents/GDG_Hack2Skill/ML-Models-GDG/CropRecommendation/Crop_recommendation.csv")  # Update with actual dataset path

# Encode categorical labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Feature selection
X = df.drop(columns=['label'])  # Features
y = df['label']  # Target variable

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
best_model = LogisticRegression(C = 10, max_iter = 1000)  
best_model.fit(X_train, y_train)

def recommend_crop(input_data):
    input_df = pd.DataFrame(input_data)
    prediction = best_model.predict(input_df)[0]
    return label_encoder.inverse_transform([prediction])[0]

def get_input():
    N = float(input("Enter Nitrogen content: "))
    P = float(input("Enter Phosphorous content: "))
    K = float(input("Enter Potassium content: "))
    temperature = float(input("Enter temperature: "))
    humidity = float(input("Enter humidity: "))
    ph = float(input("Enter pH value: "))
    rainfall = float(input("Enter rainfall: "))
    input_data = {
        "N": N, 
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }
    return [input_data]

if __name__ == "__main__":
    input_data = get_input()
    recommended_crop = recommend_crop(input_data)
    print(f"Recommended Crop: {recommended_crop}")
