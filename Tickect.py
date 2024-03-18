import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('dataset.csv')

# User input
location = input("Enter preferred location: ")
min_fee = float(input("Enter minimum fee you're ready to pay: "))
max_fee = float(input("Enter maximum fee you're ready to pay: "))
prioritize_experience = input("Do you want to prioritize years of experience? (yes/no): ")

# Filter dataset based on user input
filtered_data = data[(data['fee'] >= min_fee) & (data['fee'] <= max_fee) & (data['location'] == location)]
if prioritize_experience.lower() == 'yes':
    filtered_data = filtered_data.sort_values(by='years_of_experience', ascending=False)

if filtered_data.empty:
    print("No lawyers found matching your criteria.")
else:
    # Recommendation
    recommended_lawyer = filtered_data.iloc[0]
    print("Recommended Lawyer:")
    print("Name:", recommended_lawyer['name'])
    print("Location:", recommended_lawyer['location'])
    print("Fee:", recommended_lawyer['fee'])
    print("Years of Experience:", recommended_lawyer['years_of_experience'])
    print("Time Required:", recommended_lawyer['time_required'])
    print("Case Complexity Score:", recommended_lawyer['case_complexity_score'])
