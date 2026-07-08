# model.py

import pickle

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load Iris dataset
iris = load_iris()

# Features and Target
X = iris.data
y = iris.target

# Create Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train Model
model.fit(X, y)

# Save Model
with open("iris_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("=" * 40)
print("Model Trained Successfully")
print("iris_model.pkl created successfully")
print("=" * 40)