# train_knn.py
from reimbursement_model_knn import Model

print("--- Starting KNN Model Training and Artifact Generation ---")

# Create an instance of the model
reimbursement_engine = Model()

# Call the training method to load data and save the artifacts
reimbursement_engine.train_and_save()

print("--- Process Complete. `knn_model.pkl` is ready. ---")