
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.model_selection import GridSearchCV

# read data
X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train =  pd.read_csv("data/processed_data/y_train.csv").values.ravel()

# GridSearch with cross validation and RandomForest model
model = RandomForestRegressor(random_state = 42)
param_grid = {
	'n_estimators': [100, 200],
	'max_depth': [None, 30, 60],
	'min_samples_split': [2, 4],
	'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# save best params
with open("models/best_params.pkl", "wb") as f:
    pickle.dump(grid_search.best_params_, f)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
