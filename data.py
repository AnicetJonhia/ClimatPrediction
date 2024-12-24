import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Function for preprocessing the data
def preprocess_data(data):
    # Create a 'time' column from year, month, day, hour, and minute
    data['time'] = pd.to_datetime(data[['year', 'month', 'day', 'hour', 'minute']])

    # Extract datetime features
    data['year'] = data['time'].dt.year
    data['month'] = data['time'].dt.month
    data['day'] = data['time'].dt.day
    data['hour'] = data['time'].dt.hour
    data['minute'] = data['time'].dt.minute
    data['dayofweek'] = data['time'].dt.dayofweek  # 0 = Monday, 6 = Sunday
    data['is_weekend'] = data['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)  # 1 if weekend, 0 if weekday

    # Drop the original 'time' column after extracting features (optional)
    data = data.drop(columns=['time'], errors='ignore')

    return data


# Load your dataset (ensure to replace the path with your actual data file)
train_data = pd.read_csv('train.csv')  # Replace with your actual training data
test_data = pd.read_csv('test.csv')  # Replace with your actual test data

# Preprocess the data
train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# Define features and target for training
X_train = train_data.drop(
    columns=['Storm_NosyBe_1h', 'Storm_NosyBe_3h'] + (['storm_id'] if 'storm_id' in train_data.columns else []))  # Remove target columns and storm_id if it exists
y_train_1h = train_data['Storm_NosyBe_1h']
y_train_3h = train_data['Storm_NosyBe_3h']

# Train the model for 1-hour prediction using regression
model_1h = RandomForestRegressor(n_estimators=100, random_state=42)
model_1h.fit(X_train, y_train_1h)

# Train the model for 3-hour prediction using regression
model_3h = RandomForestRegressor(n_estimators=100, random_state=42)
model_3h.fit(X_train, y_train_3h)

# Prepare the test data
X_test = test_data.drop(
    columns=['storm_id'] + (['Storm_NosyBe_1h', 'Storm_NosyBe_3h'] if 'Storm_NosyBe_1h' in test_data.columns else []))  # Only drop 'storm_id'

# Generate predictions for both 1-hour and 3-hour forecasts
test_data['Storm_NosyBe_1h'] = model_1h.predict(X_test)  # Predicted probabilities for 1 hour
test_data['Storm_NosyBe_3h'] = model_3h.predict(X_test)  # Predicted probabilities for 3 hours

# Prepare the submission file in the required format
submission = test_data[['storm_id', 'Storm_NosyBe_1h', 'Storm_NosyBe_3h']] if 'storm_id' in test_data.columns else test_data[['Storm_NosyBe_1h', 'Storm_NosyBe_3h']]

# Save the submission file
submission.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully.")
