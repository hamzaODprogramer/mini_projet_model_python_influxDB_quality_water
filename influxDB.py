import os
import pandas as pd
from influxdb_client import InfluxDBClient, Point
from datetime import datetime, timedelta
from influxdb_client.client.write_api import SYNCHRONOUS
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree
from matplotlib import pyplot as plt
import joblib
import yaml


class InfluxDB:
    def __init__(self, token, org, url, bucket, measurement, csv , nb_rows):
        self.token = token
        self.org = org
        self.url = url
        self.bucket = bucket
        self.write_api = None
        self.write_client = None
        self.measurement = measurement
        self.csv = csv
        self.nb_rows = int(nb_rows)
        self.config = self.load_config()  # Call the load_config function

        new_config = {
            "token": token,
            "url": url,
            "org": org,
            "bucket": bucket,
            "measurement": measurement
        }

        for key, value in new_config.items():
            # Use nested dictionary access if necessary
            data_ref = self.config
            key_parts = key.split('.')
            for key_part in key_parts[:-1]:
                data_ref = data_ref.get(key_part)
            if data_ref is not None:
                data_ref[key_parts[-1]] = value

        with open("config.yaml", 'w') as f:
            yaml.safe_dump(self.config, f, default_flow_style=False)

    def load_config(self):
        with open('config.yaml', 'r') as file:
            return yaml.safe_load(file)

    def getConnection(self):
        self.write_client = InfluxDBClient(url=self.url, token=self.token, org=self.org , timeout=900_000)
        self.write_api = self.write_client.write_api(write_options=SYNCHRONOUS)

    def import_CSV_DB(self,chunk_size=100):
        """
        Reads data from a CSV file, converts timestamps, creates InfluxDB data points,
        and writes them to the specified bucket using batch writing for a specified number of rows.

        Parameters:
            num_rows (int): Number of rows to import.
            chunk_size (int): Number of rows to write in each batch.

        Returns:
            str: Message indicating successful data import.
        """
        num_rows= self.nb_rows

        # Read the CSV file
        df = pd.read_csv(self.csv)

        # Convert timestamps to nanoseconds since epoch
        # df['timestamp'] = pd.to_datetime(df['Sample Date'] + ' ' + df['Sample Time'])
        # df['timestamp'] = df['timestamp'].values.astype(np.int64) * 10**9

        # Create data points and write in batches
        total_rows = min(len(df), num_rows)
        start_idx = 0

        while start_idx < total_rows:
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk_df = df.iloc[start_idx:end_idx]

            points = []
            current_date = datetime.strptime("2020-09-01", "%Y-%m-%d")

            for index, row in chunk_df.iterrows():
                point_date = current_date.strftime("%Y-%m-%d")
                point = Point(self.measurement) \
                    .time(datetime.strptime(point_date, '%Y-%m-%d')) \
                    .tag('Sample Number', row['Sample Number']) \
                    .tag('Sample Site', row['Sample Site']) \
                    .field('Sample class', row['Sample class']) \
                    .tag('Residual Free Chlorine (mg/L)', row['Residual Free Chlorine (mg/L)']) \
                    .tag('Turbidity (NTU)', row['Turbidity (NTU)']) \
                    .tag('Fluoride (mg/L)', row['Fluoride (mg/L)']) \
                    .tag('Coliform (Quanti-Tray) (MPN /100mL)', row['Coliform (Quanti-Tray) (MPN /100mL)']) \
                    .tag('E.coli(Quanti-Tray) (MPN/100mL)', row['E.coli(Quanti-Tray) (MPN/100mL)'])
                points.append(point)

                # Increment current_date
                current_date += timedelta(days=1)

            # Write data to InfluxDB
            self.write_api.write(bucket=self.bucket, record=points)

            # Move to the next chunk
            start_idx += chunk_size

        return "yes"



    def import_CSV_DB2(self):
        """
        Reads data from a CSV file, converts timestamps, creates InfluxDB data points,
        and writes them to the specified bucket.

        Returns:
            str: Message indicating successful data import.
        """

        # Read the CSV file
        df = pd.read_csv(self.csv)

        # Convert timestamps to nanoseconds since epoch
        '''df['timestamp'] = pd.to_datetime(df['Sample Date'] + ' ' + df['Sample Time'])
        df['timestamp'] = df['timestamp'].values.astype(np.int64) * 10**9'''

        # Create data points
        points = []
        current_date = datetime.strptime("2020-09-01", "%Y-%m-%d")

        for index, row in df.head(500).iterrows():
            point_date = current_date.strftime("%Y-%m-%d")
            point = Point(self.measurement) \
                .time(datetime.strptime(point_date, '%Y-%m-%d')) \
                .tag('Sample Number', row['Sample Number']) \
                .tag('Sample Site', row['Sample Site']) \
                .field('Sample class', row['Sample class']) \
                .tag('Residual Free Chlorine (mg/L)', row['Residual Free Chlorine (mg/L)']) \
                .tag('Turbidity (NTU)', row['Turbidity (NTU)']) \
                .tag('Fluoride (mg/L)', row['Fluoride (mg/L)']) \
                .tag('Coliform (Quanti-Tray) (MPN /100mL)', row['Coliform (Quanti-Tray) (MPN /100mL)']) \
                .tag('E.coli(Quanti-Tray) (MPN/100mL)', row['E.coli(Quanti-Tray) (MPN/100mL)'])
            points.append(point)
            
            # Increment current_date
            current_date += timedelta(days=1)

        # Write data to InfluxDB
        self.write_api.write(bucket=self.bucket, record=points)

        return "yes"


    def GenerateModel(self) : 
        query = f"""
            from(bucket: "{self.bucket}")
            |> range(start: 0)  
            |> filter(fn: (r) => r["_measurement"] == "{self.measurement}")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        """

        result = self.write_client.query_api().query(query=query)

        data = []
        for table in result:
            for record in table.records:
                data.append(record.values)
        
        data = pd.DataFrame(data)
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
        #unique_classes = sorted(target.unique())
        unique_classes = sorted(target.unique())


        # Update the target_names parameter with the actual unique classes
        #classification_rep = classification_report(y_test, predictions, target_names=unique_classes)
        classification_rep = classification_report(y_test, predictions, labels=unique_classes, target_names=unique_classes)
        print("Classification Report:")
        print(classification_rep)

        # Feature Importance Analysis
        #feature_names = X_train.columns
        #feature_importance = pd.DataFrame(clf.feature_importances_, index=feature_names, columns=['Importance'])
        #features_selected = feature_importance[feature_importance['Importance'] > 0].index

        # Visualize Feature Importance
        #feature_importance.head(10).plot(kind='bar', title='Top 10 Feature Importance')

        # Visualize Decision Tree
        #fig = plt.figure(figsize=(25, 20))
        #_ = tree.plot_tree(clf, feature_names=feature_names, class_names=['Fair', 'Poor', 'Very Good', 'Excellent', 'Good'], filled=True, fontsize=12)

        # Additional: Decision Path Visualization
        sparse = clf.decision_path(X_test).toarray()[:101]
        # plt.figure(figsize=(20, 20))
        # plt.spy(sparse, markersize=5)
        # plt.show()

    def displayAllDataInTerminal(self):
        """
        Retrieves all data from the specified InfluxDB bucket and displays it in the terminal.
        """
        query = f"""
            from(bucket: "{self.bucket}")
            |> range(start: 0)  
            |> filter(fn: (r) => r["_measurement"] == "{self.measurement}")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        """

        result = self.write_client.query_api().query(query=query)

        data = []
        for table in result:
            for record in table.records:
                data.append(record.values)

        if not data:
            print("No data found.")
            return

        # Convert data to DataFrame for better display
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        df = pd.DataFrame(data)

        # Display the data
        print(df)

    # Replace the existing get_Chlorine_Time_data method in InfluxDB class
    def get_Chlorine_Time_data(self):
        query = f"""
            from(bucket: "{self.bucket}")
            |> range(start: 0)  
            |> filter(fn: (r) => r["_measurement"] == "{self.measurement}")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        """

        result = self.write_client.query_api().query(query=query)

        data = []
        for table in result:
            for record in table.records:
                data.append(record.values)

        # Convert data to DataFrame for better display
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        df = pd.DataFrame(data)

        # Convert DataFrame to dictionary
        data_dict = df.to_dict(orient='records')

        return data_dict

