import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# read data
X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv").values.ravel()

# load best params found in GridSearch 
with open("models/best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

# train model with best params
model = RandomForestRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)

# save trained model 
with open("models/trained_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved.")
