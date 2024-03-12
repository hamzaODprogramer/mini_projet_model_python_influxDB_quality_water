import os
import pandas as pd
from influxdb_client import InfluxDBClient, Point, WritePrecision
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree
from matplotlib import pyplot as plt
import joblib

token = "KpWEUoKygE57n7bg5aO_CAXUvY_qb3sd56BdL1LJzRMSA9cs4N4liwPN8DyFhX9sBPfPovg_TDczAKlN99WbHw=="
org = "HamzaOD"
bucket = "testProject"

# Connect to InfluxDB
client = InfluxDBClient(url="http://localhost:8086", token=token, org=org)

# Query the data (replace with your measurement name and adjust time range)
query = f"""
from(bucket: "{bucket}")
  |> range(start: 0)  
  |> filter(fn: (r) => r["_measurement"] == "water9")
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
"""

# Execute the query and convert the result to a Pandas DataFrame
result = client.query_api().query(query)


data = []
for table in result:
    for record in table.records:
        data.append(record.values)

# Create a DataFrame from the extracted data
data = pd.DataFrame(data)

# Check available columns in the DataFrame
print("Available Columns:")
print(data.values)

feature_cols = [
    'Residual Free Chlorine (mg/L)',
    'Turbidity (NTU)',
    'Fluoride (mg/L)',
    'Coliform (Quanti-Tray) (MPN /100mL)',
    'E.coli(Quanti-Tray) (MPN/100mL)'
]

features = data[feature_cols]
target = data['Sample class']




# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Configure the Decision Tree Classifier
clf = DecisionTreeClassifier(ccp_alpha=0.01)

# Train the model
clf.fit(X_train, y_train)

# Save the trained model to a file
model_filename = 'decision_tree_model.joblib'

os.remove('decision_tree_model.joblib')

joblib.dump(clf, model_filename)

# Make Predictions
predictions = clf.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")

conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Get unique classes in your target variable
unique_classes = sorted(target.unique())

# Update the target_names parameter with the actual unique classes
classification_rep = classification_report(y_test, predictions, target_names=unique_classes)
print("Classification Report:")
print(classification_rep)

# Feature Importance Analysis
feature_names = X_train.columns
feature_importance = pd.DataFrame(clf.feature_importances_, index=feature_names, columns=['Importance'])
features_selected = feature_importance[feature_importance['Importance'] > 0].index

# Visualize Feature Importance
feature_importance.head(10).plot(kind='bar', title='Top 10 Feature Importance')

# Visualize Decision Tree
fig = plt.figure(figsize=(25, 20))
_ = tree.plot_tree(clf, feature_names=feature_names, class_names=['Fair', 'Poor', 'Very Good', 'Excellent', 'Good'], filled=True, fontsize=12)

# Additional: Decision Path Visualization
sparse = clf.decision_path(X_test).toarray()[:101]
plt.figure(figsize=(20, 20))
plt.spy(sparse, markersize=5)

