import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
fert_data = pd.read_csv("dataset\Fertilizer Prediction.csv")

# TODO : get dataset from an [api] or [local_db]
# Encode categorical columns
label_encoder = LabelEncoder()
fert_data['Soil Type'] = label_encoder.fit_transform(fert_data['Soil Type'])
fert_data['Crop Type'] = label_encoder.fit_transform(fert_data['Crop Type'])

# Encode the target column (fertilizer names to numbers)
fert_dict = {
    'Urea': 1, 'DAP': 2, '14-35-14': 3, '28-28': 4, '17-17-17': 5,
    '20-20': 6, '10-26-26': 7
}
fert_data['Fertilizer Name'] = fert_data['Fertilizer Name'].map(fert_dict)

# Split features and target
X = fert_data.drop('Fertilizer Name', axis=1)  # Features
y = fert_data['Fertilizer Name']               # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the model and scaler
joblib.dump(model, "models/fert_model.pkl")
joblib.dump(scaler, "models/fert_scaler.pkl")