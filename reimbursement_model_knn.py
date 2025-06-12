# reimbursement_model_knn.py
import pandas as pd
import numpy as np
import json
import pickle
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

class Model:
    def __init__(self):
        self.scaler = None
        self.nn_model = None
        self.historical_df = None
        self.feature_cols = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'miles_per_day', 'spending_per_day']

    def train_and_save(self, data_filepath='public_cases.json', artifact_path='knn_model.pkl'):
        """Loads data, 'trains' the KNN model, and saves all artifacts to a file."""
        print(f"Loading data from {data_filepath}...")
        
        # Load and prepare data
        try:
            with open(data_filepath, 'r') as f: data = json.load(f)
        except FileNotFoundError: return
        valid_records, valid_outputs = [], []
        for item in data:
            try:
                record = {"trip_duration_days": int(item['input']['trip_duration_days']), "miles_traveled": int(float(item['input']['miles_traveled'])), "total_receipts_amount": float(item['input']['total_receipts_amount'])}
                valid_records.append(record); valid_outputs.append(item['expected_output'])
            except (ValueError, KeyError, TypeError): continue
        
        df = pd.DataFrame(valid_records)
        df['expected_output'] = valid_outputs
        df['miles_per_day'] = (df['miles_traveled'] / df['trip_duration_days']).replace([np.inf, -np.inf], 0).fillna(0)
        df['spending_per_day'] = (df['total_receipts_amount'] / df['trip_duration_days']).replace([np.inf, -np.inf], 0).fillna(0)
        
        self.historical_df = df

        # Scale features and fit the nearest neighbor finder
        self.scaler = StandardScaler()
        self.nn_model = NearestNeighbors(n_neighbors=1)
        
        scaled_features = self.scaler.fit_transform(self.historical_df[self.feature_cols])
        self.nn_model.fit(scaled_features)
        
        # Bundle all necessary artifacts into a single dictionary
        artifacts = {
            'scaler': self.scaler,
            'nn_model': self.nn_model,
            'historical_df': self.historical_df
        }
        
        # Save the artifacts to a file
        with open(artifact_path, 'wb') as f:
            pickle.dump(artifacts, f)
        print(f"KNN Model artifacts saved to {artifact_path}")

    def load_artifacts(self, artifact_path='knn_model.pkl'):
        """Loads the pre-trained artifacts from a file."""
        try:
            with open(artifact_path, 'rb') as f:
                artifacts = pickle.load(f)
            self.scaler = artifacts['scaler']
            self.nn_model = artifacts['nn_model']
            self.historical_df = artifacts['historical_df']
            return True
        except FileNotFoundError:
            print(f"Error: Artifact file '{artifact_path}' not found.", file=sys.stderr)
            return False

    def calculate_reimbursement(self, days, miles, receipts):
        """Finds the nearest neighbor and calculates the reimbursement."""
        input_df = pd.DataFrame([{'trip_duration_days': days, 'miles_traveled': miles, 'total_receipts_amount': receipts}])
        input_df['miles_per_day'] = (input_df['miles_traveled'] / input_df['trip_duration_days']).replace([np.inf, -np.inf], 0).fillna(0)
        input_df['spending_per_day'] = (input_df['total_receipts_amount'] / input_df['trip_duration_days']).replace([np.inf, -np.inf], 0).fillna(0)

        input_scaled = self.scaler.transform(input_df[self.feature_cols])
        distances, indices = self.nn_model.kneighbors(input_scaled)
        
        neighbor_index = indices[0][0]
        neighbor = self.historical_df.iloc[neighbor_index]
        
        prediction = neighbor['expected_output']
        
        mile_delta = input_df['miles_traveled'].iloc[0] - neighbor['miles_traveled']
        prediction += mile_delta * 0.55 
        
        receipt_delta = input_df['total_receipts_amount'].iloc[0] - neighbor['total_receipts_amount']
        prediction += receipt_delta * 0.80

        return prediction