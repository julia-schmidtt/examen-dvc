import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import json

# read data
X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv").values.ravel()

# load trained model 
with open("models/trained_model.pkl", "rb") as f:
    model = pickle.load(f)

# predict and save predictions
y_pred = model.predict(X_test)

predictions_df = pd.DataFrame({
	'y_test': y_test, 
	'y_pred': y_pred
})

predictions_df.to_csv("data/predictions.csv", index=False)

# calculate and save metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

metrics = {
    "mse": mse,
    "rmse": rmse,
    "r2": r2,
    "mae": mae
}

with open("metrics/scores.json", "w") as f:
    json.dump(metrics, f)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")
print(f"MAE: {mae:.4f}")



