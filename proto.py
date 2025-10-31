# agri_green_ai_prototype.py
# Prototype: AI-Driven Crop Recommendation Model
# Author: Vignesh P
# Project: AgriGreenAI

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1️⃣ --- Sample Dataset (Soil + Weather + Crop) ---
data = {
    "N": [90, 85, 40, 100, 60, 10, 120, 55],
    "P": [42, 38, 50, 35, 60, 20, 30, 45],
    "K": [43, 20, 40, 25, 50, 10, 30, 25],
    "temperature": [22, 26, 35, 30, 25, 20, 27, 31],
    "humidity": [80, 70, 60, 75, 65, 90, 72, 68],
    "ph": [6.5, 6.8, 7.0, 5.8, 6.0, 6.9, 7.2, 6.4],
    "rainfall": [200, 180, 120, 300, 250, 100, 400, 150],
    "crop": ["rice", "wheat", "maize", "sugarcane", "cotton", "barley", "rice", "millet"]
}

df = pd.DataFrame(data)

# 2️⃣ --- Split Data ---
X = df.drop("crop", axis=1)
y = df["crop"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3️⃣ --- Train Model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4️⃣ --- Evaluate Accuracy ---
y_pred = model.predict(X_test)
print("✅ Model Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

# 5️⃣ --- Test with Sample Sensor Input ---
# (Imagine this is coming from IoT sensors)
sample_input = {
    "N": 85,
    "P": 40,
    "K": 38,
    "temperature": 28,
    "humidity": 72,
    "ph": 6.5,
    "rainfall": 210
}

sample_df = pd.DataFrame([sample_input])

predicted_crop = model.predict(sample_df)[0]
print("\n🌾 Based on given soil & weather data, the suitable crop is:", predicted_crop)
