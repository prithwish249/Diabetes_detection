import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
df = pd.read_csv('diabetes_prediction_dataset.csv')

# Preprocess the data
df = df.dropna()

# Encode categorical variables
df = pd.get_dummies(df, columns=['gender', 'smoking_history'], drop_first=True)

# Split the data into features and target variable
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Save the feature names
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'feature_names.pkl')
print('Feature names saved as feature_names.pkl')

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, 'diabetes_model.pkl')
print('Model saved as diabetes_model.pkl')

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')
print('Scaler saved as scaler.pkl')
