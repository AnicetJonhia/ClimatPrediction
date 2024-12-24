import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np


def preprocess_data(data):
  
    data['time'] = pd.to_datetime(data[['year', 'month', 'day', 'hour', 'minute']])

    data['year'] = data['time'].dt.year
    data['month'] = data['time'].dt.month
    data['day'] = data['time'].dt.day
    data['hour'] = data['time'].dt.hour
    data['minute'] = data['time'].dt.minute
    data['dayofweek'] = data['time'].dt.dayofweek 
    data['is_weekend'] = data['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)  


    data = data.drop(columns=['time'], errors='ignore')

    return data


train_data = pd.read_csv('train.csv')  
test_data = pd.read_csv('test.csv') 


train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

X_train = train_data.drop(
    columns=['Storm_NosyBe_1h', 'Storm_NosyBe_3h'] + (['storm_id'] if 'storm_id' in train_data.columns else []))  
y_train_1h = train_data['Storm_NosyBe_1h']
y_train_3h = train_data['Storm_NosyBe_3h']


model_1h = RandomForestRegressor(n_estimators=100, random_state=42)
model_1h.fit(X_train, y_train_1h)


model_3h = RandomForestRegressor(n_estimators=100, random_state=42)
model_3h.fit(X_train, y_train_3h)

X_test = test_data.drop(
    columns=['storm_id'] + (['Storm_NosyBe_1h', 'Storm_NosyBe_3h'] if 'Storm_NosyBe_1h' in test_data.columns else []))  # Only drop 'storm_id'


test_data['Storm_NosyBe_1h'] = model_1h.predict(X_test) 
test_data['Storm_NosyBe_3h'] = model_3h.predict(X_test)

submission = test_data[['storm_id', 'Storm_NosyBe_1h', 'Storm_NosyBe_3h']] if 'storm_id' in test_data.columns else test_data[['Storm_NosyBe_1h', 'Storm_NosyBe_3h']]

submission.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully.")
