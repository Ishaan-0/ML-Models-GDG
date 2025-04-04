import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
df = pd.read_csv("/Users/ishaan/Documents/GDG_Hack2Skill/ML-Models-GDG/YieldPredictionModel/Yield_Prediction_Data.csv")


# Preprocess data (handle categorical features, scaling, etc.)
# Assuming label encoding and preprocessing are already done
X = df.drop(columns=['Yield'])  # Update target column name
y = df['Yield']

encoders = {}
for col in ['Dist Name', 'Crop']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

encoders_dist = dict(zip(encoders['Dist Name'].classes_, range(len(encoders['Dist Name'].classes_))))
encoders_crop = dict(zip(encoders['Crop'].classes_, range(len(encoders['Crop'].classes_))))

def encoders_mapping(dist_name, crop_name):
    dist_encoded = encoders['Dist Name'].transform([dist_name])[0]
    crop_encoded = encoders['Crop'].transform([crop_name])[0]
    return dist_encoded, crop_encoded

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train the best model
best_model = GradientBoostingRegressor(learning_rate=0.2, max_depth=3, n_estimators=100)
best_model.fit(X_train, y_train)

# Save the model
#joblib.dump(best_model, "best_model.pkl")

def yield_prediction(input_df):
    input_df = sc.transform(input_df)   
    y_pred = best_model.predict(input_df)
    return y_pred[0]

def get_input():
    dist_name = input("Enter District Name: ").title()
    crop = input("Enter Crop: ").upper()
    input_dist, input_crop = encoders_mapping(dist_name, crop)
    input_area = float(input("Enter Area: "))
    input_production = float(input("Enter Production: "))
    return pd.DataFrame([{
        'Dist Name': input_dist,
        'Crop': input_crop,
        'Area': input_area,
        'Production': input_production
    }])

if __name__ == "main":
    input_df = get_input()
    prediction = yield_prediction(input_df)
    print(prediction)
