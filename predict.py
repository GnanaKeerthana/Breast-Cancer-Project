import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Collect user input for all features
print("Please enter values for the following 30 features:")
user_input = []
for feature in data.feature_names:
    while True:
        try:
            value = float(input(f"{feature}: "))
            user_input.append(value)
            break
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

# Prepare input for prediction
input_array = np.array(user_input).reshape(1, -1)

# Make prediction
prediction = model.predict(input_array)[0]
result_text = "Benign (non-cancerous)" if prediction == 1 else "Malignant (cancerous)"
print(f"\n🔍 Prediction Result: {result_text}")

# Prepare data for visualization
benign_avg = X[y == 1].mean()
malignant_avg = X[y == 0].mean()

df_compare = pd.DataFrame({
    'User Input': user_input,
    'Benign Avg': benign_avg,
    'Malignant Avg': malignant_avg
}, index=data.feature_names)

# Plot the first 10 features for clarity
df_compare.iloc[:10].plot(kind='bar', figsize=(12, 6))
plt.title('User Input vs Benign & Malignant Average Feature Values (First 10 Features)')
plt.ylabel('Feature Value')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.legend(loc='upper right')
plt.show()
