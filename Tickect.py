import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('dataset.csv')

# Feature selection
X = data[['fee', 'years_of_experience', 'time_required']]
y = data['shortlisted']

# Splitting the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest model training
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# XGBoost model training
xgb_classifier = xgb.XGBClassifier()
xgb_classifier.fit(X_train, y_train)

# User input
max_fee = float(input("Enter maximum fee you're ready to pay: "))
location = input("Enter preferred location: ")
prioritize_experience = input("Do you want to prioritize years of experience? (yes/no): ")
prioritize_time = input("Do you want to prioritize time required to solve the case? (yes/no): ")

# Filter dataset based on user input
filtered_data = data[(data['fee'] <= max_fee) & (data['location'] == location)]

if prioritize_experience.lower() == 'yes':
    filtered_data = filtered_data.sort_values(by='years_of_experience', ascending=False)

if prioritize_time.lower() == 'yes':
    filtered_data = filtered_data.sort_values(by='time_required')

if filtered_data.empty:
    print("No lawyers found matching your criteria.")
else:
    # Extract features for recommendation
    X_recommend = filtered_data[['fee', 'years_of_experience', 'time_required']]

    # Predict using Random Forest model
    rf_predictions = rf_classifier.predict(X_recommend)
    filtered_data['rf_predictions'] = rf_predictions

    # Predict using XGBoost model
    xgb_predictions = xgb_classifier.predict(X_recommend)
    filtered_data['xgb_predictions'] = xgb_predictions

    # Combining predictions from both models
    filtered_data['combined_predictions'] = filtered_data['rf_predictions'] + filtered_data['xgb_predictions']

    # Recommending top lawyers based on combined predictions
    recommended_lawyers = filtered_data.sort_values(by='combined_predictions', ascending=False).head()
    print("Recommended Lawyers:")
    for idx, lawyer in recommended_lawyers.iterrows():
        print("Name:", lawyer['name'])
        print("Location:", lawyer['location'])
        print("Fee:", lawyer['fee'])
        print("Years of Experience:", lawyer['years_of_experience'])
        print("Time Required:", lawyer['time_required'])
        print("Case Complexity Score:", lawyer['case_complexity_score'])
        print("-----------------------------------")

