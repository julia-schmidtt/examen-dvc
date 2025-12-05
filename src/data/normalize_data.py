import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# read data
X_train = pd.read_csv('data/processed_data/X_train.csv')
X_test = pd.read_csv('data/processed_data/X_test.csv')

# StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# save scaled dataframes
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

X_train_scaled.to_csv("data/processed_data/X_train_scaled.csv", index=False)
X_test_scaled.to_csv("data/processed_data/X_test_scaled.csv", index=False)

# save scaler 
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print('X_train and X_test normalized with StandardScaler.')
